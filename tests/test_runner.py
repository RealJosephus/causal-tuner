"""Exercise the real runner with small weights and explicit memory budgets."""

import json
from pathlib import Path
import pytest
import torch
from safetensors import safe_open
from tokenizers import Tokenizer, models, pre_tokenizers
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.checkpointing.io import export_hf, latest_checkpoint
from causal_tuner.train import run


@pytest.fixture
def runner_assets(tmp_path, args, tiny_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    source = tmp_path/"model"
    initial = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32, attention="vanilla"))
    export_hf(initial, source, {"model_type": "test_fixture"})
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "one": 3, "two": 4, "three": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(source/"tokenizer.json"))
    (source/"tokenizer_config.json").write_text(json.dumps({"pad_token": "[PAD]", "eos_token": "[EOS]"}))
    (source/"chat_template.jinja").write_text("{{ messages }}")
    (source/"chat_templates").mkdir()
    (source/"chat_templates"/"tool_use.jinja").write_text("{{ tools }} {{ messages }}")
    dataset = tmp_path/"data.jsonl"
    dataset.write_text("\n".join(json.dumps({"text": t}) for t in ["one two three", "two one", "three one two", "one three two"]))
    args.gpu_memory_budget_gib = 5
    args.repo_id, args.dataset_name = str(source), str(dataset)
    return source


def test_runner_without_explicit_gpu_budget(tmp_path, args, runner_assets, capsys):
    args.gpu_memory_budget_gib = 0
    args.output_dir, args.max_steps = str(tmp_path/"automatic-budget"), 1
    run(args)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    configuration = next(record for record in records if "gpu_budget_gib" in record)
    assert configuration["gpu_budget_gib"] is None
    assert (tmp_path/"automatic-budget"/"adapter-00000001"/"adapter_model.safetensors").exists()


def test_runner_progress_is_independent_of_logging_interval(tmp_path, args, runner_assets, capsys):
    args.output_dir, args.max_steps = str(tmp_path/"progress"), 3
    args.logging_steps = 10
    run(args)
    captured = capsys.readouterr()
    for step in range(4):
        assert f"Training {step}/3" in captured.err
    records = [json.loads(line) for line in captured.out.splitlines() if line.startswith("{")]
    updates = [record for record in records if "microbatches" in record]
    assert len(updates) == 1 and updates[0]["step"] == 3
    final = updates[0]
    assert final["progress_percent"] == 100. and final["eta_seconds"] == 0.
    assert final["mean_step_seconds"] > 0
    assert final["training_seconds"] == pytest.approx(final["elapsed_seconds"]-final["compilation_seconds"])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("mode", ["steps", "stream", "underestimate", "overestimate"])
@pytest.mark.parametrize("cache", [False, True])
def test_runner_interruption_replay_and_exports(tmp_path, args, runner_assets, monkeypatch, capsys, dtype, mode, cache):
    from causal_tuner.data.pipeline import DataPipeline
    args.dtype = dtype
    args.warmup_steps = 1
    if cache:
        args.gradient_checkpointing, args.mlp_cache_budget_gib = "disabled", .001
    args.output_dir = str(tmp_path/"resumed")
    args.max_steps, args.gradient_accumulation_steps = 2, 2
    if mode != "steps":
        args.dataset_streaming, args.max_steps, args.num_train_epochs = True, -1, 2.
        count = {"stream": None, "underestimate": 2, "overestimate": 9}[mode]
        original = DataPipeline.__init__
        def initialize(self, *a, **kw):
            original(self, *a, **kw)
            self.source_count = count
        monkeypatch.setattr(DataPipeline, "__init__", initialize)
    args.save_optimizer_state, args.save_steps, args.save_total_limit = True, 1, 2
    args.logging_steps, args.track_memory = 1, True
    args.lora_train_lm_head = True
    args.stop_after_updates = 1
    run(args)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    metrics = next(record for record in records if 'microbatches' in record)
    assert metrics['input_batches'] == 2 and metrics['microbatches'] == 1
    first = json.loads((tmp_path/"resumed"/"checkpoint-00000001"/"manifest.json").read_text())
    expected_cursor = {"epoch": 0, "batch": 2, "micro_step": 2, "source_examples": 2}
    if mode in {"underestimate", "overestimate"}:
        expected_cursor["warmup_source_position"] = min(2, count)
    assert first["cursor"] == expected_cursor
    assert "implementation" not in first["signature"]
    with pytest.raises(ValueError, match="No compatible checkpoint"):
        latest_checkpoint(args.output_dir, {**first["signature"], "data": "changed"})
    assert not (tmp_path/"resumed"/"adapter-00000001").exists()
    args.stop_after_updates = None
    if mode != "steps":
        original_count, count = count, 1000
    run(args)
    if mode != "steps":
        count = original_count
    args.output_dir = str(tmp_path/"continuous")
    run(args)
    final_step = 2 if mode == "steps" else 4
    directory = tmp_path/"resumed"/f"adapter-{final_step:08d}"
    left = directory/"adapter_model.safetensors"
    right = tmp_path/"continuous"/f"adapter-{final_step:08d}"/"adapter_model.safetensors"
    with safe_open(left, framework="pt") as a, safe_open(right, framework="pt") as b:
        assert a.keys() == b.keys()
        for key in a.keys():
            torch.testing.assert_close(a.get_tensor(key), b.get_tensor(key), rtol=0, atol=0)
    final = json.loads((tmp_path/"resumed"/f"checkpoint-{final_step:08d}"/"manifest.json").read_text())
    assert final["scheduler"]["complete"] is True
    assert final["scheduler"]["plan"] == first["scheduler"]["plan"]
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_templates/tool_use.jinja"):
        assert (directory/name).read_bytes() == (runner_assets/name).read_bytes()


def test_streaming_warmup_boundary_survives_epoch_end_resume(tmp_path, args, runner_assets, monkeypatch):
    from causal_tuner.data.pipeline import DataPipeline
    original = DataPipeline.__init__
    def initialize(self, *a, **kw):
        original(self, *a, **kw)
        self.source_count = 2  # Deliberate underestimate: four records per epoch.
    monkeypatch.setattr(DataPipeline, "__init__", initialize)
    args.dataset_streaming, args.num_train_epochs = True, 2.
    args.total_batch_size, args.gradient_accumulation_steps = 2, 2
    # Repacking changes the number of device forwards without advancing the
    # logical data cursor or the streaming warmup position more than once.
    args.microbatch_token_budget = 3
    args.warmup_steps, args.save_optimizer_state, args.stop_after_updates = 1, True, 1
    args.output_dir = str(tmp_path/"resumed")
    run(args)
    saved = json.loads((tmp_path/"resumed"/"checkpoint-00000001"/"manifest.json").read_text())
    assert saved["cursor"] == {"epoch": 1, "batch": 0, "micro_step": 2, "source_examples": 0, "warmup_source_position": 2}
    args.stop_after_updates = None
    run(args)
    args.output_dir = str(tmp_path/"continuous")
    run(args)
    left = tmp_path/"resumed"/"adapter-00000002"/"adapter_model.safetensors"
    right = tmp_path/"continuous"/"adapter-00000002"/"adapter_model.safetensors"
    with safe_open(left, framework="pt") as a, safe_open(right, framework="pt") as b:
        for key in a.keys():
            torch.testing.assert_close(a.get_tensor(key), b.get_tensor(key), rtol=0, atol=0)


@pytest.mark.parametrize("adapter,save_optimizer", [(False, False), (True, False), (True, True)])
def test_completed_checkpoint_finishes_export_with_model_only(tmp_path, args, runner_assets, monkeypatch, adapter, save_optimizer):
    from causal_tuner.checkpointing import io
    args.output_dir, args.max_steps = str(tmp_path/"output"), 1
    args.lora_save_adapter_only = adapter
    args.save_optimizer_state = save_optimizer
    function = "export_adapter" if adapter else "export_hf"
    original = getattr(io, function)
    def interrupted(*a, **kw):
        raise OSError("interrupted export")
    monkeypatch.setattr(io, function, interrupted)
    with pytest.raises(OSError, match="interrupted export"):
        run(args)
    monkeypatch.setattr(io, function, original)
    restore = io.restore_checkpoint
    def model_only(path, manifest, model, optimizer, **kwargs):
        assert optimizer is None
        return restore(path, manifest, model, optimizer, **kwargs)
    monkeypatch.setattr(io, "restore_checkpoint", model_only)
    run(args)
    destination = tmp_path/"output"/("adapter-00000001" if adapter else "export-00000001")
    assert (destination/"tokenizer.json").read_bytes() == (runner_assets/"tokenizer.json").read_bytes()
    # A completed merged run is idempotent as well as an adapter run.
    monkeypatch.setattr(io, function, interrupted)
    run(args)


def test_completed_run_omits_source_metadata_and_exits_without_loading_model(tmp_path, args, tiny_config,
                                                                          tokenizer, monkeypatch, capsys):
    from causal_tuner.checkpointing import io
    from causal_tuner.train import runtime_from_args
    source = tmp_path/"base"
    base = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32,
                                       attention="vanilla"))
    io.export_hf(base, source, {"model_type": "fixture"}, tokenizer_directory=tmp_path/"tokenizer")
    dataset = tmp_path/"data.jsonl"
    dataset.write_text(json.dumps({"text": "one two three"}))
    args.repo_id, args.dataset_name = str(source), str(dataset)
    args.output_dir, args.max_steps, args.use_wandb = str(tmp_path/"output"), 1, True
    model = CausalLM(tiny_config, runtime_from_args(args))
    checkpoint = io.save_checkpoint(args.output_dir, model, None, step=1, cursor={}, signature={},
                                    scheduler={"complete": True, "plan": {"steps": 1, "batches": [1],
                                                                        "source_per_epoch": None}})
    io.export_adapter(model, Path(args.output_dir)/"adapter-00000001", base_identity=str(source))
    latest, snapshots = io.latest_checkpoint, []
    def completed(output, signature):
        assert set(signature) == {"base", "data", "arguments", "runtime", "format"}
        manifest = io.read_manifest(checkpoint)
        manifest["signature"] = {**signature, "implementation": "unused-source-metadata"}
        io.json_write(checkpoint/"manifest.json", manifest)
        snapshots.append((checkpoint/"manifest.json").read_bytes())
        return latest(output, signature)
    def forbidden(*a, **kw):
        pytest.fail("A completed run must exit without loading, training, tracking or exporting")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("causal_tuner.kernels.probe.probe_backends", lambda *a: None)
    monkeypatch.setattr(io, "latest_checkpoint", completed)
    for target in ("causal_tuner.modeling.model.CausalLM", "causal_tuner.checkpointing.huggingface.load_weights",
                   "causal_tuner.training.engine.make_optimizer", "causal_tuner.training.engine.Trainer",
                   "causal_tuner.training.tracking.start_wandb", "causal_tuner.checkpointing.io.restore_checkpoint",
                   "causal_tuner.checkpointing.io.export_hf", "causal_tuner.checkpointing.io.export_adapter"):
        monkeypatch.setattr(target, forbidden)
    run(args)
    assert capsys.readouterr().out.strip() == "The requested training and export are already complete."
    assert (checkpoint/"manifest.json").read_bytes() == snapshots[0]
