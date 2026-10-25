import copy
import json
import pytest
import torch
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.train import runtime_from_args
from causal_tuner.training.engine import Trainer, make_optimizer, learning_rate
from causal_tuner.checkpointing.io import save_checkpoint, latest_checkpoint, restore_checkpoint, export_hf
from causal_tuner.checkpointing.huggingface import load_weights
from conftest import batch


def test_padding_embedding_preserves_loaded_values_and_suppresses_lookup_gradient(tiny_config):
    model = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32,
                                         attention="vanilla"))
    embedding = model.model.embed_tokens
    assert not embedding.weight[tiny_config.pad_token_id].any()
    with torch.no_grad():
        embedding.weight[tiny_config.pad_token_id].fill_(3.)
    ids = torch.tensor([[tiny_config.pad_token_id, 3]])
    actual = embedding(ids)
    assert torch.all(actual[0, 0] == 3.)
    actual.sum().backward()
    assert not embedding.weight.grad[tiny_config.pad_token_id].any()
    assert torch.all(embedding.weight.grad[3] == 1.)


def test_adam_decay_excludes_biases_and_norms_and_warmup_starts_at_zero(args, tiny_config):
    model = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32))
    args.weight_decay, args.learning_rate, args.warmup_steps = .1, .01, 2
    optimizer = make_optimizer(model, args)
    original = {name: param.detach().clone() for name, param in model.named_parameters()}
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    for name, param in model.named_parameters():
        factor = 1-args.learning_rate*args.weight_decay if param.ndim > 1 else 1
        torch.testing.assert_close(param, original[name]*factor, rtol=0, atol=0)
    assert learning_rate(args, 0, 10) == 0
    assert learning_rate(args, 1, 10) == .005
    assert learning_rate(args, 2, 10) == .01
    assert learning_rate(args, 10, 10) == args.learning_rate_end


@pytest.mark.parametrize("weights", [(.05, .15), (0., 0.), (.1, .9), (1., 7.)])
def test_accumulation_matches_single_weighted_batch(args, tiny_config, weights):
    args.max_grad_norm = 0
    args.lora_train_lm_head = True
    args.lora_train_embed = True
    model = CausalLM(tiny_config, runtime_from_args(args))
    control = copy.deepcopy(model)
    a, b = weights
    micro = [batch([[2, 3, 4]], [[0., 0., a]]), batch([[5, 6, 7]], [[0., b, 0.]])]
    together = {k: torch.cat([m[k] for m in micro]) for k in micro[0]}
    one, two = make_optimizer(model, args), make_optimizer(control, args)
    Trainer(model, one, args, device="cpu").update(micro, step=0, total_steps=2)
    Trainer(control, two, args, device="cpu").update([together], step=0, total_steps=2)
    for (name, p), (_, q) in zip(model.named_parameters(), control.named_parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(p, q, atol=1e-7, rtol=2e-5)
        else:
            assert p not in one.state


@pytest.mark.parametrize("embed,head", [(False, False), (True, False), (False, True), (True, True)])
def test_lora_tables_and_second_step(args, tiny_config, embed, head):
    args.lora_train_embed, args.lora_train_lm_head = embed, head
    model = CausalLM(tiny_config, runtime_from_args(args))
    optimizer = make_optimizer(model, args)
    trainer = Trainer(model, optimizer, args, device="cpu")
    before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    for step in range(2):
        trainer.update([batch([[2, 4, 6, 8]])], step=step, total_steps=2)
    for name, p in model.named_parameters():
        if "lora_A" in name:
            assert p.grad.abs().sum() > 0, name
        if not p.requires_grad:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert p.grad is None and p not in optimizer.state
    assert model.model.embed_tokens.weight.requires_grad == embed
    assert model.lm_head.weight.requires_grad == head


def test_atomic_checkpoint_exact_resume_and_hf_merge(tmp_path, args, tiny_config):
    args.lora_train_embed = args.lora_train_lm_head = True
    args.learning_rate = .003
    model = CausalLM(tiny_config, runtime_from_args(args))
    initial = copy.deepcopy(model)
    optimizer = make_optimizer(model, args)
    trainer = Trainer(model, optimizer, args, device="cpu")
    data = batch([[2, 3, 4, 5]])
    trainer.update([data], step=0, total_steps=2)
    signature = {"base": "test", "data": "fixture"}
    path = save_checkpoint(tmp_path, model, optimizer, step=1, cursor={"epoch": 0, "batch": 1},
                           signature=signature, scheduler={"step": 1, "total": 2}, save_optimizer=True)
    (tmp_path/".incomplete-dead").mkdir()
    (tmp_path/"checkpoint-99999999").mkdir()
    found = latest_checkpoint(tmp_path, signature)
    assert found[0] == path
    trainer.update([data], step=1, total_steps=2)
    resumed_optimizer = make_optimizer(initial, args)
    step, cursor = restore_checkpoint(*found, initial, resumed_optimizer)
    Trainer(initial, resumed_optimizer, args, device="cpu").update([data], step=step, total_steps=2)
    for p, q in zip(model.parameters(), initial.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(ValueError, match="compatible"):
        latest_checkpoint(tmp_path, {"different": True})
    export_hf(model, tmp_path/"merged", {"model_type": "fixture"})
    full = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32, attention="vanilla"), device="meta")
    load_weights(full, tmp_path/"merged", device="cpu")
    torch.testing.assert_close(full(data), model(data), rtol=2e-5, atol=2e-5)


def test_missing_optimizer_cannot_resume(tmp_path, args, tiny_config):
    model = CausalLM(tiny_config, runtime_from_args(args))
    optimizer = make_optimizer(model, args)
    save_checkpoint(tmp_path, model, optimizer, step=0, cursor={}, signature={}, scheduler={})
    with pytest.raises(ValueError, match="optimizer"):
        restore_checkpoint(*latest_checkpoint(tmp_path, {}), model, optimizer)


def test_peft_export_keys_and_streamed_values(tmp_path, args, tiny_config):
    from causal_tuner.checkpointing.io import export_adapter
    from safetensors import safe_open
    args.lora_train_embed = args.lora_train_lm_head = True
    model = CausalLM(tiny_config, runtime_from_args(args))
    export_adapter(model, tmp_path/"adapter", base_identity="test-base")
    config = json.loads((tmp_path/"adapter"/"adapter_config.json").read_text())
    assert config["lora_alpha"] == config["r"] == 4
    assert set(config["modules_to_save"]) == {"embed_tokens", "lm_head"}
    with safe_open(tmp_path/"adapter"/"adapter_model.safetensors", framework="pt") as handle:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                key = "base_model.model."+name+(".weight" if "lora_" in name else "")
                torch.testing.assert_close(handle.get_tensor(key), parameter)


def test_full_small_model_training_with_frozen_embedding(args, tiny_config):
    args.lora = False
    args.lora_save_adapter_only = False
    args.frozen_components = ("embed",)
    model = CausalLM(tiny_config, runtime_from_args(args))
    optimizer = make_optimizer(model, args)
    metrics = Trainer(model, optimizer, args, device="cpu").update([batch([[2, 3, 4]])], step=0, total_steps=2)
    assert metrics["effective_weight"] == 2
    assert model.model.embed_tokens.weight.grad is None


@pytest.mark.parametrize("total_weight", [0., .2, 1.])
def test_normalization_denominator_against_direct_objective(args, tiny_config, total_weight):
    from causal_tuner.data.segments import prediction_targets
    args.learning_rate = 0
    args.max_grad_norm = 0
    args.lora_train_lm_head = True
    model = CausalLM(tiny_config, runtime_from_args(args))
    control = copy.deepcopy(model)
    data = batch([[2, 3, 4]], [[0., total_weight, 0.]])
    targets, weights = prediction_targets(data)
    logits = torch.nn.functional.linear(control(data), control.lm_head.weight)
    loss = (torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")*weights.flatten()).sum()/max(total_weight, 1.)
    loss.backward()
    Trainer(model, make_optimizer(model, args), args, device="cpu").update([data], step=0, total_steps=1)
    for p, q in zip(model.parameters(), control.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)


def test_reference_bf16_persistent_state(args, tiny_config):
    args.precision_recipe = "reference_bf16"
    args.dtype = args.param_dtype = "bfloat16"
    model = CausalLM(tiny_config, runtime_from_args(args))
    optimizer = make_optimizer(model, args)
    Trainer(model, optimizer, args, device="cpu").update([batch([[2, 3, 4]])], step=0, total_steps=1)
    for p in model.parameters():
        if p.requires_grad:
            assert p.dtype == p.grad.dtype == optimizer.state[p]["exp_avg"].dtype == optimizer.state[p]["exp_avg_sq"].dtype == torch.bfloat16


@pytest.mark.parametrize("scheduler", ["constant", "linear", "cosine"])
def test_short_schedule_and_absolute_end_rate(args, scheduler):
    from causal_tuner.training.engine import learning_rate
    args.scheduler, args.learning_rate, args.learning_rate_end = scheduler, .01, .002
    args.warmup_steps = 1
    assert learning_rate(args, 0, 1) == .01
    with pytest.raises(ValueError, match="horizon"):
        learning_rate(args, 0, 2, warmup=2)
    with pytest.raises(ValueError, match="at least one"):
        learning_rate(args, 0, 0)
    assert learning_rate(args, 0, None) == 0
    assert learning_rate(args, 100, None) == .01
    assert learning_rate(args, 10, 10) == (.01 if scheduler == "constant" else .002)


@pytest.mark.parametrize("field", ["base", "data", "arguments", "runtime"])
def test_checkpoint_matches_current_fields_without_rewriting_saved_metadata(tmp_path, args, tiny_config, field):
    model = CausalLM(tiny_config, runtime_from_args(args))
    signature = {"base": "same-base", "data": "same-data", "arguments": {"batch": 1}, "runtime": {"torch": "same"}}
    saved = {**signature, "implementation": "unused-source-metadata"}
    path = save_checkpoint(tmp_path, model, None, step=1, cursor={}, signature=saved, scheduler={"complete": True})
    before = (path/"manifest.json").read_bytes()
    assert latest_checkpoint(tmp_path, signature)[0] == path
    assert (path/"manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="No compatible checkpoint"):
        latest_checkpoint(tmp_path, {**signature, field: "changed"})


