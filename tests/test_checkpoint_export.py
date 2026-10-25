"""Incomplete checkpoint export leaves continuation and tracking untouched."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from conftest import batch
from test_hub_storage import hub
from causal_tuner.checkpointing.io import (copy_tokenizer, export_hf, read_manifest, restore_checkpoint,
                                          save_checkpoint, latest_checkpoint)
from causal_tuner.checkpointing.huggingface import base_identity, load_weights
from causal_tuner.export import parser, run
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.train import runtime_from_args
from causal_tuner.training.engine import Trainer, make_optimizer


@pytest.fixture
def saved_run(tmp_path, args, tiny_config, tokenizer):
    assets = tmp_path/"base"
    base = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32, attention="vanilla"))
    export_hf(base, assets, {"model_type": "fixture"})
    copy_tokenizer(tmp_path/"tokenizer", assets)
    args.repo_id, args.dataset_name = str(assets), "unavailable-training-data"
    args.max_steps, args.save_optimizer_state, args.use_wandb = 3, True, True
    model = CausalLM(tiny_config, runtime_from_args(args), device="meta")
    load_weights(model, assets, device="cpu")
    optimizer = make_optimizer(model, args)
    Trainer(model, optimizer, args, device="cpu").update([batch([[2, 3, 4]])], step=0, total_steps=3)
    signature = {"base": base_identity(assets), "data": "unavailable", "implementation": "original-source",
                 "arguments": {k: v for k, v in vars(args).items() if k not in {"token", "frozen_components"}}}
    options = dict(step=1, cursor={"epoch": 0, "batch": 1, "micro_step": 1, "source_examples": 1},
                   signature=signature, scheduler={"step": 1, "complete": False,
                       "plan": {"steps": 3, "batches": None, "source_per_epoch": None}}, save_optimizer=True,
                   tracking={"wandb": {"id": "same-run", "project": "project", "entity": "team"}},
                   export={"base_revision": None, "processor_revision": None, "tokenizer": tokenizer.signature})
    return model, optimizer, options


def verify_export(directory, kind, model, tiny_config):
    if kind == "adapter":
        actual = load_file(directory/"adapter_model.safetensors")
        for name, value in model.named_parameters():
            if value.requires_grad:
                key = "base_model.model."+name+(".weight" if name.endswith(("lora_A", "lora_B")) else "")
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        assert json.loads((directory/"adapter_config.json").read_text())["peft_type"] == "LORA"
    else:
        assert [path.name for path in directory.glob("*.safetensors")] == ["model.safetensors"]
        assert not (directory/"model.safetensors.index.json").exists()
        merged = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32,
                                              attention="vanilla"), device="meta")
        load_weights(merged, directory, device="cpu")
        torch.testing.assert_close(merged(batch([[2, 3, 4]])), model(batch([[2, 3, 4]])), rtol=2e-5, atol=2e-5)
    assert (directory/"tokenizer.json").is_file()


@pytest.mark.parametrize("kind", ["adapter", "model"])
def test_local_incomplete_export_then_exact_continuation(saved_run, tmp_path, args, tiny_config, monkeypatch, kind):
    from causal_tuner.training import engine, tracking
    from causal_tuner.data.pipeline import DataPipeline
    model, optimizer, options = saved_run
    checkpoint = save_checkpoint(tmp_path/"training", model, optimizer, **options)
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
    destination = tmp_path/kind
    export_args = parser().parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(destination), "--format", kind])
    def forbidden(*a, **kw):
        pytest.fail("Export must not initialize or update training state")
    with monkeypatch.context() as patch:
        patch.setattr(engine, "make_optimizer", forbidden)
        patch.setattr(Trainer, "update", forbidden)
        patch.setattr(tracking, "start_wandb", forbidden)
        patch.setattr(DataPipeline, "__init__", forbidden)
        patch.setattr(torch, "set_rng_state", forbidden)
        patch.setattr(torch.cuda, "set_rng_state", forbidden)
        assert run(export_args) == destination
    verify_export(destination, kind, model, tiny_config)
    assert {p.name: p.read_bytes() for p in checkpoint.iterdir()} == before
    signature = {key: value for key, value in options["signature"].items() if key != "implementation"}
    found = latest_checkpoint(tmp_path/"training", signature)
    assert found[1]["scheduler"]["complete"] is False
    assert found[1]["tracking"] == options["tracking"]
    with pytest.raises(ValueError, match="compatible"):
        latest_checkpoint(tmp_path/"training", {**signature, "data": "changed"})
    resumed = copy.deepcopy(model)
    resumed_optimizer = make_optimizer(resumed, args)
    step, cursor = restore_checkpoint(*found, resumed, resumed_optimizer)
    assert step == 1 and cursor == options["cursor"]
    for current, current_optimizer in ((model, optimizer), (resumed, resumed_optimizer)):
        Trainer(current, current_optimizer, args, device="cpu").update([batch([[4, 5, 6]])], step=step, total_steps=3)
    for left, right in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if left.requires_grad:
            for name, value in optimizer.state[left].items():
                torch.testing.assert_close(value, resumed_optimizer.state[right][name], rtol=0, atol=0)


@pytest.mark.parametrize("revision", ["latest", "first"])
@pytest.mark.parametrize("kind", ["adapter", "model"])
def test_hub_export_is_read_only_and_pins_one_commit(hub, saved_run, args, tmp_path, tiny_config, revision, kind):
    from causal_tuner.checkpointing.hub import HubCheckpoint
    model, optimizer, options = saved_run
    args.hub_model_repo_id = "owner/export-source"
    remote = HubCheckpoint(args)
    remote.save(model, optimizer, **options)
    first_sha, first_model = remote.repo.revision, copy.deepcopy(model)
    Trainer(model, optimizer, args, device="cpu").update([batch([[4, 5, 6]])], step=1, total_steps=3)
    remote.save(model, optimizer, **{**options, "step": 2})
    destination = tmp_path/"export"
    flags = ["--checkpoint", args.hub_model_repo_id, "--output-dir", str(destination), "--format", kind]
    if revision == "first":
        flags += ["--checkpoint-revision", first_sha]
    before, events = copy.deepcopy(hub.repos), list(hub.events)
    run(parser().parse_args(flags))
    verify_export(destination, kind, first_model if revision == "first" else model, tiny_config)
    assert hub.repos == before and hub.events == events
    expected_sha = first_sha if revision == "first" else remote.repo.revision
    assert all(sha == expected_sha for _, sha, _ in hub.downloads)
    assert [name for name, _, _ in hub.downloads].count("checkpoint/state.safetensors") == 1
    assert all(not path.exists() for _, _, path in hub.downloads)


def test_export_without_optimizer_still_cannot_resume_training(saved_run, args, tmp_path):
    model, optimizer, options = saved_run
    checkpoint = save_checkpoint(tmp_path/"training", model, optimizer, **{**options, "save_optimizer": False})
    run(parser().parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path/"adapter")]))
    with pytest.raises(ValueError, match="optimizer"):
        restore_checkpoint(checkpoint, read_manifest(checkpoint), model, optimizer)


def test_export_rejects_changed_base_and_output_inside_checkpoint(saved_run, args, tmp_path):
    model, optimizer, options = saved_run
    checkpoint = save_checkpoint(tmp_path/"training", model, optimizer, **options)
    flags = ["--checkpoint", str(checkpoint)]
    with pytest.raises(ValueError, match="outside"):
        run(parser().parse_args([*flags, "--output-dir", str(checkpoint/"adapter")]))
    manifest = read_manifest(checkpoint)
    manifest["signature"]["base"] = {"changed": True}
    (checkpoint/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="base weights"):
        run(parser().parse_args([*flags, "--output-dir", str(tmp_path/"adapter")]))


def test_export_help_does_not_import_torch():
    code = "import sys; from causal_tuner.export import parser; parser().format_help(); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]/"src")})
    assert result.returncode == 0, result.stderr


def test_full_checkpoint_uses_saved_frozen_parameters_and_defaults_to_model(saved_run, args, tiny_config, tmp_path):
    _, _, options = saved_run
    args.lora, args.lora_save_adapter_only = False, False
    args.frozen_parameters, args.frozen_components = "embed", ("embed",)
    model = CausalLM(tiny_config, runtime_from_args(args), device="meta")
    load_weights(model, args.repo_id, device="cpu")
    optimizer = make_optimizer(model, args)
    Trainer(model, optimizer, args, device="cpu").update([batch([[2, 3, 4]])], step=0, total_steps=3)
    options["signature"]["arguments"] = {k: v for k, v in vars(args).items() if k not in {"token", "frozen_components"}}
    checkpoint = save_checkpoint(tmp_path/"training", model, optimizer, **options)
    destination = tmp_path/"full"
    run(parser().parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(destination)]))
    verify_export(destination, "model", model, tiny_config)
    with pytest.raises(ValueError, match="LoRA"):
        run(parser().parse_args(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path/"adapter"), "--format", "adapter"]))
