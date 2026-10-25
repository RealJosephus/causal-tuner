"""Real tensor/Datasets serialization against an in-memory Hub protocol fixture."""

import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from causal_tuner.checkpointing.hub import HubCheckpoint
from causal_tuner.publishing import HubRepository, json_bytes
from test_runner import runner_assets


class HubFixture:
    def __init__(self, directory):
        self.directory, self.repos, self.blobs = directory, {}, {}
        self.events, self.downloads, self.buffers = [], [], []
        self.fail_settings = self.fail_upload = self.fail_commit = None
        self.ignore_settings = False

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        key = repo_type, repo_id
        if key not in self.repos:
            self.repos[key] = dict(private=private, gated=False, sha="0"*40, versions={"0"*40: {}})
        self.events.append(("create", key))

    def update_repo_settings(self, repo_id, *, repo_type, private, gated):
        self.events.append(("settings", (repo_type, repo_id), private, gated))
        if self.fail_settings:
            raise self.fail_settings
        if not self.ignore_settings:
            self.repos[repo_type, repo_id].update(private=private, gated=gated)

    def repo_info(self, repo_id, *, repo_type, revision=None):
        repo = self.repos[repo_type, repo_id]
        sha = repo["sha"] if revision in (None, "main") else revision
        assert sha in repo["versions"]
        return SimpleNamespace(id=repo_id, private=repo["private"], gated=repo["gated"], sha=sha)

    def dataset_info(self, repo_id, *, revision=None):
        repo = self.repos["dataset", repo_id]
        return SimpleNamespace(sha=revision or repo["sha"])

    def content(self, operation):
        source = operation.path_or_fileobj
        if isinstance(source, bytes):
            return source
        if isinstance(source, (str, Path)):
            return Path(source).read_bytes()
        with operation.as_file() as handle:
            handle.seek(0)
            blocks = []
            while block := handle.read(4*1024**2):
                self.buffers.append(len(block))
                blocks.append(block)
            return b"".join(blocks)

    def preupload_lfs_files(self, repo_id, *, repo_type, additions, free_memory=True, **kwargs):
        if self.fail_upload:
            raise self.fail_upload
        for operation in additions:
            data = self.content(operation)
            self.blobs[operation.upload_info.sha256] = data
            self.events.append(("preupload", (repo_type, repo_id), operation.path_in_repo))
            assert free_memory
            operation.path_or_fileobj = b""
            operation._is_uploaded = True

    def put_version(self, repo_id, repo_type, changes):
        repo = self.repos[repo_type, repo_id]
        files = dict(repo["versions"][repo["sha"]])
        for name, data in changes.items():
            if data is None:
                files.pop(name, None)
            else:
                files[name] = data
        sha = f"{len(repo['versions']):040x}"
        repo["versions"][sha], repo["sha"] = files, sha
        return sha

    def create_commit(self, repo_id, *, repo_type, operations, parent_commit, commit_message, **kwargs):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        repo = self.repos[repo_type, repo_id]
        assert parent_commit == repo["sha"]
        failure, self.fail_commit = self.fail_commit, None
        if failure == "before":
            raise TimeoutError("request did not commit")
        changes = {}
        for operation in operations:
            if isinstance(operation, CommitOperationDelete):
                changes[operation.path_in_repo] = None
            else:
                assert isinstance(operation, CommitOperationAdd)
                changes[operation.path_in_repo] = (self.blobs[operation.upload_info.sha256]
                    if operation._is_uploaded else self.content(operation))
        sha = self.put_version(repo_id, repo_type, changes)
        self.events.append(("commit", (repo_type, repo_id), sha))
        return SimpleNamespace(oid=sha)

    def download(self, repo_id, filename, *, repo_type, revision, token=None, local_dir=None):
        from huggingface_hub.errors import EntryNotFoundError
        files = self.repos[repo_type, repo_id]["versions"][revision]
        if filename not in files:
            raise EntryNotFoundError(filename)
        root = Path(local_dir) if local_dir else self.directory/"cache"/repo_type/repo_id/revision
        path = root/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        self.downloads.append((filename, revision, path))
        return str(path)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    import huggingface_hub
    fixture = HubFixture(tmp_path)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: fixture)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fixture.download)
    monkeypatch.setattr("causal_tuner.hub.time.sleep", lambda delay: None)
    return fixture


@pytest.fixture
def native_dataset_hub(hub, monkeypatch):
    """Replace network storage only; keep Dataset.push_to_hub and Parquet real."""
    import datasets.arrow_dataset as arrow
    from fsspec.implementations.memory import MemoryFileSystem
    def filesystem(**kwargs):
        fs = MemoryFileSystem()
        for (kind, name), repo in hub.repos.items():
            for sha, files in repo["versions"].items():
                root = f"datasets/{name}@{sha}"
                fs.makedirs(root, exist_ok=True)
                for filename, data in files.items():
                    fs.pipe_file(root+"/"+filename, data)
        return fs
    monkeypatch.setattr(arrow, "HfApi", lambda **kwargs: hub)
    monkeypatch.setattr(arrow, "HfFileSystem", filesystem)
    return hub


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_public_manual_access_is_applied_before_upload(hub, args, repo_type):
    repo = HubRepository("owner/test", repo_type, args)
    repo.preupload("weights.safetensors", b"payload")
    assert hub.repos[repo_type, "owner/test"]["private"] is False
    assert hub.repos[repo_type, "owner/test"]["gated"] == "manual"
    assert [event[0] for event in hub.events] == ["create", "settings", "preupload"]


@pytest.mark.parametrize("failure", ["rejected", "not_applied"])
def test_access_configuration_failure_cannot_upload(hub, args, failure):
    if failure == "rejected":
        hub.fail_settings = PermissionError("not allowed")
    else:
        hub.ignore_settings = True
    with pytest.raises((PermissionError, ValueError)):
        HubRepository("owner/test", "model", args)
    assert not hub.blobs and not hub.buffers
    assert hub.repos["model", "owner/test"]["private"] is True


def model_state(args, tiny_config):
    from causal_tuner.modeling.model import CausalLM
    from causal_tuner.train import runtime_from_args
    from causal_tuner.training.engine import Trainer, make_optimizer
    from conftest import batch
    model = CausalLM(tiny_config, runtime_from_args(args))
    initial = copy.deepcopy(model)
    optimizer = make_optimizer(model, args)
    Trainer(model, optimizer, args, device="cpu").update([batch([[2, 3, 4]])], step=0, total_steps=3)
    return model, optimizer, initial


def save_options(step=1):
    return dict(step=step, cursor=dict(epoch=0, batch=step, micro_step=step, source_examples=step),
                signature={"fixture": "same-code-base-data"}, scheduler={"step": step, "complete": False,
                "plan": {"steps": 3, "batches": [3], "source_per_epoch": None}}, chunk_bytes=1024)


def test_streamed_checkpoint_resumes_model_adam_rng_and_cursor_exactly(hub, args, tiny_config):
    from causal_tuner.training.engine import Trainer, make_optimizer
    from conftest import batch
    args.hub_model_repo_id = "owner/run"
    model, optimizer, initial = model_state(args, tiny_config)
    remote = HubCheckpoint(args)
    assert remote.latest(save_options()["signature"]) is None
    remote.save(model, optimizer, **save_options())
    expected_random = random.random(), np.random.rand(), torch.rand(3)
    assert max(hub.buffers) < 1400
    assert len(hub.repos["model", "owner/run"]["versions"]) == 2
    Trainer(model, optimizer, args, device="cpu").update([batch([[4, 5, 6]])], step=1, total_steps=3)
    restored = HubCheckpoint(args)
    assert restored.latest(save_options()["signature"])[1]["step"] == 1
    resumed_optimizer = make_optimizer(initial, args)
    step, cursor = restored.restore(initial, resumed_optimizer)
    assert step == 1 and cursor == save_options()["cursor"]
    assert random.random() == expected_random[0] and np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2], rtol=0, atol=0)
    assert all(not path.exists() for _, _, path in hub.downloads)
    Trainer(initial, resumed_optimizer, args, device="cpu").update([batch([[4, 5, 6]])], step=step, total_steps=3)
    for a, b in zip(model.parameters(), initial.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if a.requires_grad:
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[a][key], resumed_optimizer.state[b][key], rtol=0, atol=0)


def test_checkpoint_git_versions_keep_history_and_replace_same_paths(hub, args, tiny_config):
    args.hub_model_repo_id = "owner/run"
    model, optimizer, _ = model_state(args, tiny_config)
    remote = HubCheckpoint(args)
    remote.save(model, optimizer, **save_options())
    first_sha = remote.repo.revision
    first_tree = dict(hub.repos["model", "owner/run"]["versions"][first_sha])
    for param in model.parameters():
        if param.requires_grad:
            param.data.add_(.25)
    options = save_options(2)
    options["chunk_bytes"] = 4096
    remote.save(model, optimizer, **options)
    repo = hub.repos["model", "owner/run"]
    assert len(repo["versions"]) == 3
    assert repo["versions"][first_sha] == first_tree
    latest = repo["versions"][repo["sha"]]
    assert set(latest) == set(first_tree) == {"checkpoint/state.safetensors", "checkpoint/manifest.json"}
    assert json.loads(latest["checkpoint/manifest.json"])["step"] == 2
    assert all(name.startswith("checkpoint/") for name in latest)


def test_transient_commit_failure_uses_existing_retry(hub, args, tiny_config):
    args.hub_model_repo_id = "owner/run"
    model, optimizer, _ = model_state(args, tiny_config)
    remote = HubCheckpoint(args)
    hub.fail_commit = "before"
    remote.save(model, optimizer, **save_options())
    assert len(hub.repos["model", "owner/run"]["versions"]) == 2
    assert len([event for event in hub.events if event[0] == "commit"]) == 1


def test_remote_manifest_and_chunk_validation(hub, args, tiny_config):
    args.hub_model_repo_id = "owner/run"
    model, optimizer, initial = model_state(args, tiny_config)
    remote = HubCheckpoint(args)
    remote.save(model, optimizer, **save_options())
    with pytest.raises(ValueError, match="compatible"):
        HubCheckpoint(args).latest({"changed": True})
    name = next(iter(remote.manifest["files"]))
    hub.put_version("owner/run", "model", {"checkpoint/"+name: b"damaged"})
    restored = HubCheckpoint(args)
    restored.latest(save_options()["signature"])
    with pytest.raises(ValueError, match="checksum"):
        restored.restore(initial, None, require_optimizer=False)
    damaged = copy.deepcopy(remote.manifest)
    damaged["files"]["../outside"] = "checksum"
    hub.put_version("owner/run", "model", {"checkpoint/manifest.json": json_bytes(damaged)})
    with pytest.raises(ValueError, match="outside"):
        HubCheckpoint(args).latest(save_options()["signature"])


def test_final_adapter_shares_repo_without_removing_checkpoint(hub, args, tiny_config, tmp_path):
    from causal_tuner.checkpointing.io import export_adapter
    args.hub_model_repo_id = "owner/run"
    model, optimizer, _ = model_state(args, tiny_config)
    remote = HubCheckpoint(args)
    remote.save(model, optimizer, **save_options())
    first = dict(hub.repos["model", "owner/run"]["versions"][remote.repo.revision])
    directory = tmp_path/"adapter"
    export_adapter(model, directory, base_identity="owner/base")
    assert not remote.export_complete(1, save_options()["signature"], "adapter")
    remote.publish_export(directory, step=1, signature=save_options()["signature"], kind="adapter")
    assert remote.export_complete(1, save_options()["signature"], "adapter")
    latest = hub.repos["model", "owner/run"]["versions"][remote.repo.revision]
    assert all(latest[name] == data for name, data in first.items())
    assert "adapter_model.safetensors" in latest and "adapter_config.json" in latest


@pytest.mark.parametrize("field", ["base", "data", "arguments", "runtime"])
def test_checkpoint_and_export_match_current_fields_without_updating_saved_metadata(hub, args, tiny_config, tmp_path, field):
    from causal_tuner.checkpointing.io import export_adapter
    args.hub_model_repo_id = "owner/run"
    model, optimizer, _ = model_state(args, tiny_config)
    signature = {"base": "same-base", "data": "same-data", "arguments": {"batch": 1}, "runtime": {"torch": "same"}}
    saved = {**signature, "implementation": "unused-source-metadata"}
    remote = HubCheckpoint(args)
    remote.save(model, optimizer, **{**save_options(), "signature": saved})
    directory = tmp_path/"adapter"
    export_adapter(model, directory, base_identity="owner/base")
    remote.publish_export(directory, step=1, signature=saved, kind="adapter")
    before, events = copy.deepcopy(hub.repos), list(hub.events)
    assert remote.latest(signature)[1]["step"] == 1
    assert remote.export_complete(1, signature, "adapter")
    changed = {**signature, field: "changed"}
    assert not remote.export_complete(1, changed, "adapter")
    with pytest.raises(ValueError, match="No compatible checkpoint"):
        remote.latest(changed)
    assert hub.repos == before and hub.events == events


@pytest.mark.parametrize("streaming", [False, True])
def test_prepare_pushes_native_dataset_and_loads_final_commit(native_dataset_hub, args, tokenizer, tmp_path, monkeypatch, streaming):
    from datasets import Dataset, load_dataset
    from causal_tuner.data.pipeline import DataPipeline
    from causal_tuner.prepare import prepare_dataset
    hub = native_dataset_hub
    source = tmp_path/"raw.jsonl"
    source.write_text('\n'.join(json.dumps({"text": row}) for row in ("one two", "one three two", "two three")))
    args.dataset_name, args.output_dir = str(source), str(tmp_path/"unused-output")
    args.hub_dataset_repo_id, args.max_sequence_length = "owner/data", 32
    expected = [{k: list(v) for k, v in row.items() if not k.startswith("_")}
                for row in DataPipeline(args, tokenizer, 97).examples(0)]

    native_push = Dataset.push_to_hub
    def push(self, *positional, **options):
        assert "max_shard_size" not in options and "num_shards" not in options
        assert self.column_names == ["input_ids"]
        return native_push(self, *positional, **options)
    monkeypatch.setattr(Dataset, "push_to_hub", push)

    def load(repo_id, config, *, revision, split, token, streaming):
        parquet = tmp_path/"library-read.parquet"
        files = hub.repos["dataset", repo_id]["versions"][revision]
        parquet.write_bytes(next(data for name, data in files.items() if name.endswith(".parquet")))
        return load_dataset("parquet", data_files=str(parquet), split=split, streaming=streaming,
                            cache_dir=str(tmp_path/"datasets-cache"))

    monkeypatch.setattr("datasets.load_dataset", load)
    url = prepare_dataset(args, tokenizer, 97)
    assert not Path(args.output_dir).exists()
    final_sha = url.rsplit("/", 1)[1]
    manifest = json.loads(hub.repos["dataset", "owner/data"]["versions"][final_sha]["manifest.json"])
    assert "data_revision" not in manifest and manifest["records"] == 3 and manifest["version"] == 3
    args.dataset_name, args.dataset_format, args.dataset_revision = "owner/data", "prepared", final_sha
    args.dataset_streaming = streaming
    actual = DataPipeline(args, tokenizer, 97)
    assert [{k: list(v) for k, v in row.items() if not k.startswith("_")} for row in actual.records()] == expected


@pytest.mark.parametrize("failure", [False, True])
def test_native_push_parquet_batches_respect_largest_row_bound(native_dataset_hub, args, failure):
    import io
    import pyarrow.parquet as pq
    from datasets import Dataset, config
    from causal_tuner.data.prepared import features_for, saving_batch_size
    hub = native_dataset_hub
    repo = HubRepository("owner/skewed", "dataset", args)
    columns = ["input_ids", "attention_mask", "loss_weights"]
    sizes = [4]*100+[40000]*2
    dataset = Dataset.from_dict({name: [[1]*n for n in sizes] for name in columns}, features=features_for(columns))
    previous = config.DEFAULT_MAX_BATCH_SIZE, config.MAX_ROW_GROUP_SIZE
    if failure:
        hub.fail_upload = RuntimeError("network rejected upload")
    try:
        with saving_batch_size(5, dataset=dataset):
            dataset.push_to_hub(repo.repo_id, split="train")
    except RuntimeError as exc:
        assert failure and "network rejected" in str(exc)
    else:
        assert not failure
        tree = hub.repos["dataset", repo.repo_id]
        files = tree["versions"][tree["sha"]]
        groups, values = [], []
        for name, data in files.items():
            if name.endswith(".parquet"):
                parquet = pq.ParquetFile(io.BytesIO(data))
                groups.extend(parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups))
                values.extend(parquet.read().to_pylist())
        assert groups and max(groups) <= 5 and sum(groups) == len(dataset)
        assert values == list(dataset)
    assert (config.DEFAULT_MAX_BATCH_SIZE, config.MAX_ROW_GROUP_SIZE) == previous


def test_output_cli_defaults_and_optimizer_requirement():
    from causal_tuner.arguments import parse_args, parser
    flags = ["--repo-id", "owner/base", "--dataset-name", "owner/data"]
    args = parse_args([*flags, "--hub-model-repo-id", "owner/run", "--save-optimizer-state"])
    assert args.hub_private is False and args.hub_gated == "manual"
    assert args.save_steps == 0 and args.save_total_limit == 1
    assert "--hub-dataset-repo-id" not in parser().format_help()
    assert "--hub-model-repo-id" not in parser(prepare=True).format_help()
    prepared = parse_args([*flags, "--hub-dataset-repo-id", "owner/prepared"], prepare=True)
    assert prepared.hub_dataset_repo_id == "owner/prepared"
    with pytest.raises(SystemExit):
        parse_args([*flags, "--hub-model-repo-id", "owner/run"])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_runner_repeated_command_resumes_hub_then_finishes_once(hub, args, runner_assets, tmp_path, dtype):
    from causal_tuner.train import run
    from safetensors.torch import load
    args.dtype, args.hub_model_repo_id = dtype, "owner/run"
    args.output_dir = str(tmp_path/"no-local-checkpoints")
    args.max_steps, args.stop_after_updates = 2, 1
    args.save_optimizer_state, args.save_steps = True, 1
    run(args)
    first_sha = hub.repos["model", "owner/run"]["sha"]
    first_tree = hub.repos["model", "owner/run"]["versions"][first_sha]
    assert json.loads(first_tree["checkpoint/manifest.json"])["step"] == 1
    # The identical invocation stops after another update and completes the
    # original schedule. No local checkpoint is needed for the second run.
    run(args)
    final_sha = hub.repos["model", "owner/run"]["sha"]
    final_tree = hub.repos["model", "owner/run"]["versions"][final_sha]
    manifest = json.loads(final_tree["checkpoint/manifest.json"])
    assert manifest["scheduler"]["complete"] and manifest["step"] == 2
    assert not Path(args.output_dir).exists()
    assert final_tree["tokenizer.json"] == (runner_assets/"tokenizer.json").read_bytes()
    run(args)
    assert hub.repos["model", "owner/run"]["sha"] == final_sha
    assert len(hub.repos["model", "owner/run"]["versions"]) == 4  # initial, two checkpoints, export
    args.hub_model_repo_id, args.stop_after_updates = None, None
    args.output_dir = str(tmp_path/"continuous")
    run(args)
    expected = load((Path(args.output_dir)/"adapter-00000002"/"adapter_model.safetensors").read_bytes())
    actual = load(final_tree["adapter_model.safetensors"])
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    local_checkpoint = Path(args.output_dir)/"checkpoint-00000002"
    local_manifest = json.loads((local_checkpoint/"manifest.json").read_text())
    assert manifest["tensors"].keys() == local_manifest["tensors"].keys()
    # Restoring JSON metadata can change Adam's dictionary insertion order.
    # Compare logical tensor identities, not incidental shard file numbers.
    for name, entry in manifest["tensors"].items():
        other = local_manifest["tensors"][name]
        assert entry["shape"] == other["shape"] and entry["dtype"] == other["dtype"]
        left = load(final_tree["checkpoint/"+entry["file"]])[name]
        right = load((local_checkpoint/other["file"]).read_bytes())[name]
        torch.testing.assert_close(left, right, rtol=0, atol=0)
