"""Standalone adapter merges preserve inference, tables and source assets."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from conftest import batch
from causal_tuner.checkpointing.huggingface import load_weights
from causal_tuner.checkpointing.io import copy_tokenizer, export_adapter, export_hf
from causal_tuner.data.tokenizer import TrainingTokenizer
from causal_tuner.merge import main, parser, run
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.train import runtime_from_args


@pytest.fixture
def merge_assets(tmp_path, args, tiny_config, tokenizer, request):
    base, adapter = tmp_path/"base", tmp_path/"adapter"
    initial = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32,
                                          attention="vanilla"))
    export_hf(initial, base, {"model_type": "fixture"}, tokenizer_directory=tmp_path/"tokenizer")
    args.lora_train_embed, args.lora_train_lm_head = getattr(request, "param", (True, True))
    model = CausalLM(tiny_config, runtime_from_args(args), device="meta")
    load_weights(model, base, device="cpu")
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith("lora_B"):
                value.normal_(std=.015)
            elif value.requires_grad and not name.endswith("lora_A"):
                value.add_(torch.randn_like(value)*.01)
    export_adapter(model, adapter, base_identity=str(base), tokenizer_directory=tmp_path/"tokenizer")
    config_path = adapter/"tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] += "\n"
    config_path.write_text(json.dumps(config))
    return base, adapter, model


def options(adapter, destination, *flags):
    return parser().parse_args(["--adapter", str(adapter), "--output-dir", str(destination), *flags])


def update_config(adapter, updates):
    path = adapter/"adapter_config.json"
    config = json.loads(path.read_text())
    config.update(updates)
    path.write_text(json.dumps(config))


@pytest.mark.parametrize("merge_assets", [(False, False), (True, False), (False, True), (True, True)], indirect=True)
@pytest.mark.parametrize("alpha", [4, 8])
def test_merge_matches_adapter_inference_and_preserves_sources(merge_assets, tmp_path, tiny_config, alpha):
    base, adapter, model = merge_assets
    update_config(adapter, {"lora_alpha": alpha})
    before = {path: path.read_bytes() for source in (base, adapter) for path in source.rglob("*") if path.is_file()}
    destination = tmp_path/"merged"
    assert run(options(adapter, destination)) == destination
    assert [path.name for path in destination.glob("*.safetensors")] == ["model.safetensors"]
    assert not (destination/"model.safetensors.index.json").exists()
    merged = CausalLM(tiny_config, Runtime(rank=0, base_dtype=torch.float32, compute_dtype=torch.float32,
                                         attention="vanilla"), device="meta")
    load_weights(merged, destination, device="cpu")
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith("lora_B"):
                value.mul_(alpha/model.runtime.rank)
        data = batch([[2, 3, 4], [5, 6, 7]])
        expected = torch.nn.functional.linear(model(data), model.lm_head.weight)
        actual = torch.nn.functional.linear(merged(data), merged.lm_head.weight)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(merged.model.embed_tokens.weight, model.model.embed_tokens.weight, rtol=0, atol=0)
    torch.testing.assert_close(merged.lm_head.weight, model.lm_head.weight, rtol=0, atol=0)
    assert TrainingTokenizer(destination).signature == TrainingTokenizer(adapter).signature
    assert json.loads((destination/"export_provenance.json").read_text()) == {"merged": True, "base": "unquantized"}
    assert all(path.read_bytes() == content for path, content in before.items())


def test_base_override_output_dtype_and_base_tokenizer_fallback(merge_assets, tmp_path):
    base, adapter, model = merge_assets
    metadata = json.loads((base/"config.json").read_text())
    metadata["dtype"] = "float32"
    (base/"config.json").write_text(json.dumps(metadata))
    update_config(adapter, {"base_model_name_or_path": None})
    (adapter/"tokenizer.json").unlink()
    destination = tmp_path/"merged"
    run(options(adapter, destination, "--repo-id", str(base), "--dtype", "bfloat16"))
    name = "model.layers.0.self_attn.q_proj.weight"
    projection = model.model.layers[0].self_attn.q_proj
    expected = (projection.weight.bfloat16().float()+projection.lora_B @ projection.lora_A).bfloat16()
    actual = load_file(destination/"model.safetensors")[name]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert json.loads((destination/"config.json").read_text())["torch_dtype"] == "bfloat16"
    assert json.loads((destination/"config.json").read_text())["dtype"] == "bfloat16"
    assert TrainingTokenizer(destination).signature == TrainingTokenizer(base).signature


def test_remote_adapter_and_base_pin_separate_revisions(merge_assets, tmp_path, monkeypatch, capsys):
    import huggingface_hub
    base, adapter, _ = merge_assets
    requests, downloads = [], []
    def info(self, repo, *, revision=None):
        requests.append((repo, revision))
        return SimpleNamespace(sha="adapter-sha" if repo == "owner/adapter" else "base-sha")
    def download(**kwargs):
        downloads.append(kwargs)
        return str(adapter if kwargs["repo_id"] == "owner/adapter" else base)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    destination = tmp_path/"merged"
    main(["--adapter", "owner/adapter", "--adapter-revision", "adapter-tag", "--repo-id", "owner/base",
          "--revision", "base-tag", "--output-dir", str(destination)])
    assert capsys.readouterr().out.strip() == str(destination)
    assert requests == [("owner/adapter", "adapter-tag"), ("owner/base", "base-tag")]
    assert [call["revision"] for call in downloads] == ["adapter-sha", "base-sha"]
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"} <= set(downloads[0]["allow_patterns"])
    assert "config.json" in downloads[1]["allow_patterns"]
    assert (destination/"config.json").is_file()


@pytest.mark.parametrize("updates", [
    {"r": 0}, {"lora_alpha": float("nan")}, {"peft_type": "OTHER"}, {"bias": "all"},
    {"target_modules": ["q_proj"]}, {"modules_to_save": ["other"]}, {"use_dora": True},
    {"use_rslora": True}, {"rank_pattern": {"q_proj": 2}}, {"fan_in_fan_out": True},
])
def test_unsupported_adapter_settings_fail_before_loading_base(merge_assets, tmp_path, monkeypatch, updates):
    _, adapter, _ = merge_assets
    update_config(adapter, updates)
    def forbidden(*a, **kw):
        pytest.fail("Invalid adapters must be rejected before loading base weights")
    monkeypatch.setattr("causal_tuner.checkpointing.huggingface.load_weights", forbidden)
    destination = tmp_path/"merged"
    with pytest.raises(ValueError):
        run(options(adapter, destination))
    assert not destination.exists()


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "dtype"])
def test_invalid_adapter_tensors_fail_before_loading_base(merge_assets, tmp_path, monkeypatch, problem):
    _, adapter, _ = merge_assets
    path = adapter/"adapter_model.safetensors"
    tensors = load_file(path)
    key = next(iter(tensors))
    if problem == "missing":
        tensors.pop(key)
    elif problem == "unexpected":
        tensors["unexpected"] = torch.ones(1)
    elif problem == "shape":
        tensors[key] = torch.ones(1)
    else:
        tensors[key] = tensors[key].to(torch.int32)
    save_file(tensors, path)
    def forbidden(*a, **kw):
        pytest.fail("Invalid adapters must be rejected before loading base weights")
    monkeypatch.setattr("causal_tuner.checkpointing.huggingface.load_weights", forbidden)
    destination = tmp_path/"merged"
    with pytest.raises(ValueError, match="Adapter parameter keys mismatch|Invalid adapter tensor"):
        run(options(adapter, destination))
    assert not destination.exists()


@pytest.mark.parametrize("source", ["adapter", "base"])
def test_merge_rejects_destination_inside_sources(merge_assets, source):
    base, adapter, _ = merge_assets
    destination = (base if source == "base" else adapter)/"merged"
    with pytest.raises(ValueError, match="outside"):
        run(options(adapter, destination))
    assert not destination.exists()


def test_merge_rejects_existing_destination(merge_assets):
    base, adapter, _ = merge_assets
    before = (base/"config.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(options(adapter, base))
    assert (base/"config.json").read_bytes() == before


def test_merge_requires_base_location(merge_assets, tmp_path):
    _, adapter, _ = merge_assets
    update_config(adapter, {"base_model_name_or_path": None})
    with pytest.raises(ValueError, match="--repo-id"):
        run(options(adapter, tmp_path/"merged"))


@pytest.mark.parametrize("flags", [
    ("--host-memory-budget-gib", "0"), ("--host-memory-budget-gib", "73"),
    ("--host-memory-budget-gib", "nan"), ("--hf-retry-initial-delay", "0"),
    ("--hf-retry-initial-delay", "61"), ("--hf-retry-max-delay", "inf"),
])
def test_invalid_limits_fail_before_resolving_assets(tmp_path, monkeypatch, flags):
    def forbidden(*a, **kw):
        pytest.fail("Invalid options must fail before accessing the Hub")
    monkeypatch.setattr("causal_tuner.checkpointing.huggingface.resolve_assets", forbidden)
    with pytest.raises(ValueError, match="budget|delays"):
        run(options("owner/adapter", tmp_path/"merged", *flags))


def test_merge_help_does_not_import_torch():
    code = "import sys; from causal_tuner.merge import parser; parser().format_help(); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]/"src")})
    assert result.returncode == 0, result.stderr


def test_merge_module_command(merge_assets, tmp_path):
    _, adapter, _ = merge_assets
    destination = tmp_path/"merged-cli"
    result = subprocess.run([sys.executable, "-m", "causal_tuner.merge", "--adapter", str(adapter),
                             "--output-dir", str(destination)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]/"src")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(destination)
    assert (destination/"model.safetensors").is_file()
    assert not (destination/"model.safetensors.index.json").exists()
    assert TrainingTokenizer(destination).signature == TrainingTokenizer(adapter).signature
