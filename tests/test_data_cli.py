import json
import subprocess
import sys
import pytest
import torch
from causal_tuner.arguments import parse_args
from causal_tuner.data.pipeline import DataPipeline, raw_record, normalize, pack_group, bounded_shuffle
from causal_tuner.data.segments import prediction_targets
from causal_tuner.prepare import prepare_dataset


def test_cli_aliases_and_explicit_rejections():
    args = parse_args(["--repo_id", "x", "--dataset-name", "y", "--lora", "YES", "--no-shuffle", "--lora-train-lm-head", "1"])
    assert args.lora and args.lora_train_lm_head and not args.shuffle
    assert args.total_batch_size == 1 and args.lora_rank == 256 and args.lora_save_adapter_only
    assert args.loss_implementation == "fused" and args.attn_mechanism == "flash_attention_2"
    assert args.gradient_checkpointing == "auto" and args.activation_offload == "auto"
    assert args.microbatch_token_budget == 12288
    for flags in (["--distributed"], ["--async-dispatch-steps", "2"], ["--block-size-q", "128"],
                  ["--learning-rate", "nan"], ["--gradient-checkpointing", "disabled", "--remat-group-size", "2"],
                  ["--lora-train-embed-and-lm-head", "--no-lora-train-embed"],
                  ["--rmsnorm-backward", "exact"], ["--cut-filter-eps", "none"],
                  ["--loss-implementation", "auto"], ["--loss-implementation", "chunked"],
                  ["--attention-sparse-backend", "reference"], ["--cut-chunk-size", "4"]):
        with pytest.raises(SystemExit):
            parse_args(["--repo-id", "x", "--dataset-name", "y", *flags])
    code = "import sys; from causal_tuner.arguments import parse_args; parse_args(['--repo-id','x','--dataset-name','y']); assert 'torch' not in sys.modules and 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env={**__import__('os').environ, "PYTHONPATH": "src"})


@pytest.mark.parametrize("flags", [
    ["--max-sequence-length", "1"], ["--seed", "-1"], ["--seed", str(2**32)],
    ["--learning-rate", "0"], ["--learning-rate-end", "1"], ["--num-train-epochs", "0"],
    ["--assistant-only-loss"], ["--last-assistant-only-loss"],
    ["--endprompt-enable", "--endprompt-prompt-loss-weight", "0", "--endprompt-context-loss-weight", "0"],
    ["--loss-implementation", "cut", "--lora-train-lm-head"], ["--loss-implementation", "cut", "--no-lora"],
    ["--mlp-cache-budget-gib", "-1"], ["--mlp-cache-budget-gib", "1", "--gradient-checkpointing", "enabled"],
    ["--microbatch-token-budget", "-1"], ["--attention-cache-budget-gib", "-1"],
    ["--attention-cache-budget-gib", "1", "--remat-group-size", "2"],
    ["--mlp-cache-budget-gib", "1", "--gradient-checkpointing", "disabled", "--no-lora"],
])
def test_invalid_training_options_fail_before_loading(flags):
    with pytest.raises(SystemExit):
        parse_args(["--repo-id", "x", "--dataset-name", "y", *flags])


def test_prepare_exposes_only_processing_options_and_flat_mlp_default():
    from causal_tuner.arguments import parser
    flags = ["--repo-id", "x", "--dataset-name", "y"]
    prepared = parse_args(flags, prepare=True)
    assert not prepared.shuffle and prepared.dataset_format == "raw"
    help_text = parser(prepare=True).format_help()
    for excluded in ("--learning-rate", "--lora", "--activation-offload", "--stop-after-updates", "--dataset-format", "--mlp-cache-budget-gib"):
        assert excluded not in help_text
        with pytest.raises(SystemExit):
            parse_args([*flags, excluded, "1"], prepare=True)
    assert parse_args([*flags, "--total-batch-size", "8"]).mlp_chunk_size is None
    assert parse_args(flags).mlp_chunk_size is None
    assert parse_args([*flags, "--total-batch-size", "8", "--mlp-chunk-size", "0"]).mlp_chunk_size == 0
    assert parse_args([*flags, "--max-steps", "2", "--num-train-epochs", "0"]).max_steps == 2
    cached = parse_args([*flags, "--mlp-cache-budget-gib", "1.5", "--gradient-checkpointing", "disabled"])
    from causal_tuner.train import runtime_from_args
    assert runtime_from_args(cached).mlp_cache_budget_bytes == int(1.5*1024**3)
    assert parse_args([*flags, "--loss-implementation", "cut", "--no-lora", "--frozen-parameters", "lm_head"]).loss_implementation == "cut"


@pytest.mark.parametrize("prepare", [False, True])
def test_endprompt_logical_range_is_not_fixed(prepare, tokenizer):
    args = parse_args(["--repo-id", "x", "--dataset-name", "y", "--max-sequence-length", "9",
                       "--endprompt-enable", "--endprompt-logical-length", "4194304",
                       "--endprompt-prompts", "anchor"], prepare=prepare)
    record = normalize(raw_record({"text": "one two"}, tokenizer, args, 0), 9, 97)
    assert record["position_ids"] == [0, 1, 4194303]


def test_raw_and_message_weights_and_endprompt(args, tokenizer):
    raw = raw_record({"text": "one two three"}, tokenizer, args, 0)
    assert raw["input_ids"] == [5, 6, 7]
    args.dataset_text_field = "messages"
    args.assistant_only_loss = True
    row = {"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two", "loss": .2},
                        {"role": "user", "content": "three"}, {"role": "assistant", "content": "one", "loss": False}]}
    args.max_sequence_length = 30
    encoded = raw_record(row, tokenizer, args, 0)
    assert sum(encoded["loss_weights"]) == pytest.approx(.4)
    args.assistant_only_loss, args.last_assistant_only_loss = False, True
    assert raw_record(row, tokenizer, args, 0) is None
    args.last_assistant_only_loss = False
    args.dataset_text_field = "text"
    args.endprompt_enable = True
    args.endprompt_prompts = "anchor"
    args.endprompt_logical_length, args.endprompt_logical_length_min = 2000000, 1999000
    first = raw_record({"text": "one two"}, tokenizer, args, 123)
    second = raw_record({"text": "one two"}, tokenizer, args, 123)
    assert first == second and first["position_ids"][2] >= 1998999
    assert first["loss_weights"] == [1., 1., .1]
    assert len(first["input_ids"]) == 3


def test_template_compilation_reuse_and_token_crossing_generation_boundaries(tokenizer):
    from causal_tuner.data.tokenizer import compiled_template
    compiled_template.cache_clear()
    tokenizer.template = "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}"
    messages = [{"role": "assistant", "content": "one", "weight": .25},
                {"role": "assistant", "content": "two", "weight": .75}]
    ids, weights = tokenizer.messages(messages, assistant_only=True)
    assert len(ids) == 1 and weights == [.75]
    messages[-1]["weight"] = .5
    assert tokenizer.messages(messages, assistant_only=True)[1] == [.5]
    info = compiled_template.cache_info()
    assert info.misses == 1 and info.hits == 1


@pytest.mark.parametrize("logical_position", [1999998, 4194302])
def test_packing_preserves_internal_boundaries_and_positions(logical_position):
    one = normalize({"input_ids": [2, 3, 4, 5], "attention_mask": [1, 0, 1, 1], "segment_ids": [0, 0, 0, 0],
                     "position_ids": [0, 0, logical_position, logical_position+1]}, 8, 97)
    two = normalize({"input_ids": [6, 7]}, 8, 97)
    packed = pack_group([one, two], 8, 0)[0]
    assert list(packed["input_ids"]) == [2, 3, 4, 5, 6, 7]
    assert list(packed["segment_ids"]) == [1, 0, 2, 2, 3, 3]
    assert list(packed["position_ids"]) == [0, 0, logical_position, logical_position+1, 0, 1]
    tensors = {k: torch.tensor([v]) for k, v in packed.items()}
    _, weights = prediction_targets(tensors)
    assert weights.sum() == 2


def test_prepare_replay_and_compact_tail(tmp_path, args, tokenizer):
    source = tmp_path/"source.jsonl"
    source.write_text('\n'.join(json.dumps({"text": text}) for text in ["one two", "three two", "one three"]))
    args.dataset_name = str(source)
    args.output_dir = str(tmp_path/"prepared")
    args.total_batch_size = 2
    raw = DataPipeline(args, tokenizer, 97)
    original = list(raw.batches())
    assert len(original) == 2
    assert original[0]["input_ids"].shape == (1, 4)
    assert original[1]["input_ids"].shape == (1, 2)
    prepare_dataset(args, tokenizer, 97)
    args.dataset_format, args.dataset_name = "prepared", args.output_dir
    prepared = DataPipeline(args, tokenizer, 97)
    actual = list(prepared.batches())
    for x, y in zip(original, actual):
        for key in x:
            if torch.is_tensor(x[key]):
                torch.testing.assert_close(x[key], y[key])
    remainder = list(prepared.batches(skip=1))
    torch.testing.assert_close(remainder[0]["input_ids"], actual[1]["input_ids"])
    args.endprompt_enable = True
    with pytest.raises(ValueError, match="signature"):
        DataPipeline(args, tokenizer, 97)


def test_data_budget_and_invalid_prepared_groups():
    with pytest.raises(MemoryError, match="record"):
        list(bounded_shuffle([{"text": "x"*1000}], 20, 100, 0))
    with pytest.raises(ValueError, match="complete intervals"):
        normalize({"input_ids": [1, 2, 3], "attention_end": [0, 5, 5]}, 3, 97)
    with pytest.raises(ValueError, match="finite"):
        normalize({"input_ids": [1, 2], "loss_weights": [1, float("nan")]}, 3, 97)
    with pytest.raises(ValueError, match="non-negative"):
        normalize({"input_ids": [1, 2], "position_ids": [0, -1]}, 3, 97)
@pytest.mark.parametrize("expression,options", [
    ("tojson", {}),
    ("tojson(ensure_ascii=True, indent=2, separators=(',', ': '), sort_keys=True)",
     dict(ensure_ascii=True, indent=2, separators=(",", ": "), sort_keys=True)),
    ("tojson(False, 1, (',', ':'), True)", dict(indent=1, separators=(",", ":"), sort_keys=True)),
])
def test_chat_tojson_matches_hf_defaults_and_arguments(expression, options):
    from causal_tuner.data.tokenizer import compiled_template
    tools = [{"description": "\u67e5\u8be2\u5929\u6c14", "city": "\u53f0\u5317", "html": "<>&'"}]
    expected = json.dumps(tools, **{**dict(ensure_ascii=False, indent=None, separators=None, sort_keys=False), **options})
    actual = compiled_template("{{ tools | "+expression+" }}").render(tools=tools)
    assert actual == expected
    assert "<>&'" in actual

