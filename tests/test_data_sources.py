import json
from types import SimpleNamespace
import pytest
import torch
from causal_tuner import hub
from causal_tuner.arguments import parse_args
from causal_tuner.data.pipeline import DataPipeline
from causal_tuner.prepare import prepare_dataset
from causal_tuner.train import training_plan, streaming_schedule, streaming_position
from causal_tuner.training.engine import learning_rate


@pytest.mark.parametrize("extension", ["json", "jsonl", "csv", "tsv", "parquet"])
def test_local_formats_share_processing(extension, tmp_path, args, tokenizer):
    rows = [{"text": "one two"}, {"text": "two three"}, {"text": "three one"}]
    path = tmp_path/f"data.{extension}"
    if extension == "json":
        pytest.importorskip("ijson")
        path.write_text(json.dumps(rows))
    elif extension == "jsonl":
        path.write_text("\n".join(map(json.dumps, rows)))
    elif extension in {"csv", "tsv"}:
        path.write_text("text\n"+"\n".join(r["text"] for r in rows))
    else:
        pa = pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), path)
    args.dataset_name = str(path)
    batches = list(DataPipeline(args, tokenizer, 97).batches())
    assert [b["input_ids"][0, :2].tolist() for b in batches] == [[5, 6], [6, 7], [7, 5]]
    assert sum(b["_source_count"] for b in batches) == len(rows)


def test_source_counts_survive_filter_shuffle_pack_and_prefetch(tmp_path, args, tokenizer):
    rows = ["", "one two", "one", "two three", "", "three two", ""]
    path = tmp_path/"data.jsonl"
    path.write_text("\n".join(json.dumps({"text": row}) for row in rows))
    args.dataset_name, args.packing, args.shuffle, args.prefetch_batches = str(path), True, True, 2
    pipeline = DataPipeline(args, tokenizer, 97)
    original = list(pipeline.batches())
    replayed = list(pipeline.batches())
    assert sum(b["_source_count"] for b in original) == len(rows)
    for a, b in zip(original, replayed):
        assert a["_source_count"] == b["_source_count"]
        torch.testing.assert_close(a["input_ids"], b["input_ids"])


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_prepared_uses_datasets_at_one_pinned_revision(tmp_path, args, tokenizer, monkeypatch, streaming):
    import datasets
    import huggingface_hub
    path = tmp_path/"data.jsonl"
    path.write_text(json.dumps({"text": "one two"}))
    args.dataset_name, args.output_dir = str(path), str(tmp_path/"prepared")
    directory = prepare_dataset(args, tokenizer, 97)
    stored = datasets.load_from_disk(str(directory))
    calls = []
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda *a, **k: SimpleNamespace(sha="immutable"))
    def download(repo, filename, **kwargs):
        calls.append((filename, kwargs["revision"]))
        return str(directory/filename)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    loaded = []
    def load(repo, config, **kwargs):
        loaded.append((repo, config, kwargs))
        return stored.to_iterable_dataset() if kwargs["streaming"] else stored
    monkeypatch.setattr(datasets, "load_dataset", load)
    args.dataset_name, args.dataset_format = "fixture/remote", "prepared"
    args.dataset_streaming = streaming
    pipeline = DataPipeline(args, tokenizer, 97)
    assert calls == [("manifest.json", "immutable")]
    assert loaded == [("fixture/remote", None, dict(split="train", revision="immutable", token=None, streaming=streaming))]
    assert pipeline.source_count == 1
    assert len(list(pipeline.batches())) == 1
    assert calls == [("manifest.json", "immutable")]
    import shutil
    damaged = tmp_path/"damaged"
    shutil.copytree(directory, damaged)
    args.dataset_name = str(damaged)
    next(damaged.glob("*.arrow")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        DataPipeline(args, tokenizer, 97)


def test_retry_backoff_exceeds_five_attempts_and_rejects_permanent_errors(monkeypatch):
    sleeps, attempts = [], []
    monkeypatch.setattr(hub.time, "sleep", sleeps.append)
    def call():
        attempts.append(1)
        if len(attempts) < 8:
            raise ConnectionError("offline")
        return 42
    assert hub.retry(call, initial_delay=1, max_delay=4) == 42
    assert sleeps == [1, 2, 4, 4, 4, 4, 4]
    for error in (FileNotFoundError(), ValueError(), PermissionError()):
        assert not hub.transient(error)
    import httpx
    for status in (401, 403, 404, 429, 503):
        error = httpx.HTTPStatusError("response", request=httpx.Request("GET", "https://example.invalid"),
                                    response=httpx.Response(status))
        error.__cause__ = ConnectionError()
        assert hub.transient(error) == (status in (429, 503))
    # Exercise the transport selected by the installed Hub SDK as well.
    # Hub releases use requests, httpx, or httpx2 depending on their version.
    import importlib
    from huggingface_hub.utils import get_session
    transport = importlib.import_module(type(get_session()).__module__.split(".")[0])
    assert hub.transient(transport.ReadTimeout("interrupted download"))


@pytest.mark.parametrize("changed", [False, True])
def test_disconnected_stream_replays_without_duplicates(changed):
    attempts = 0
    def factory():
        nonlocal attempts
        attempts += 1
        yield {"text": "changed" if changed and attempts > 1 else "one"}
        if attempts == 1:
            raise ConnectionError("interrupted")
        yield {"text": "two"}
    rows = hub.replay(factory, fields=("text",))
    assert next(rows) == {"text": "one"}
    if changed:
        with pytest.raises(ValueError, match="prefix"):
            list(rows)
    else:
        assert list(rows) == [{"text": "two"}]
    assert attempts == 2


@pytest.mark.parametrize("count", [None, 3, 100])
def test_streaming_plan_never_scans_and_schedule_caps_metadata(args, count):
    args.dataset_streaming, args.num_train_epochs = True, 2.
    def forbidden(*args):
        raise AssertionError("Streaming must not be scanned to plan epochs")
    pipeline = SimpleNamespace(source_count=count, batches=forbidden)
    plan = training_plan(args, pipeline)
    assert plan["steps"] is None and plan["source_per_epoch"] == count
    cursor = dict(epoch=0, source_examples=(count or 3)*2)
    schedule = streaming_schedule(args, plan, cursor, 2)
    if count is None:
        assert learning_rate(args, *schedule) == args.learning_rate
    else:
        assert schedule[0] == 2 and schedule[1] is None
        assert 0 < schedule[3] < .5
        cursor["source_examples"] *= 10
        assert streaming_schedule(args, plan, cursor, 2) == schedule
    with pytest.raises(SystemExit):
        parse_args(["--repo-id", "x", "--dataset-name", "y", "--dataset-streaming", "--num-train-epochs", "1.5"])


@pytest.mark.parametrize("scheduler", ["linear", "cosine", "constant"])
@pytest.mark.parametrize("counts", [[1, 2, 3], [10, 50, 99], [50, 200, 900]])
def test_streaming_warmup_uses_updates_and_commits_decay_origin(args, scheduler, counts):
    import math
    args.scheduler, args.warmup_steps, args.num_train_epochs = scheduler, 3, 2.
    plan = {"source_per_epoch": 100}
    cursor = {"epoch": 0, "source_examples": 0}
    for step, count in enumerate(counts):
        assert learning_rate(args, *streaming_schedule(args, plan, cursor, step)) == pytest.approx(args.learning_rate*step/3)
        cursor["source_examples"] = count
    cursor["warmup_source_position"] = streaming_position(plan, cursor)
    assert learning_rate(args, *streaming_schedule(args, plan, cursor, 3)) == args.learning_rate
    cursor.update(epoch=1, source_examples=50)
    fraction = (150-cursor["warmup_source_position"])/(210-cursor["warmup_source_position"])
    factor = 1. if scheduler == "constant" else 1-fraction if scheduler == "linear" else (1+math.cos(math.pi*fraction))/2
    expected = args.learning_rate_end+factor*(args.learning_rate-args.learning_rate_end)
    restored = json.loads(json.dumps(cursor))
    assert learning_rate(args, *streaming_schedule(args, plan, restored, 4)) == pytest.approx(expected)


def test_streaming_warmup_is_not_compared_to_record_horizon(args):
    args.warmup_steps, args.num_train_epochs = 100, 1.
    schedule = streaming_schedule(args, {"source_per_epoch": 1}, {"epoch": 0, "source_examples": 1}, 10)
    assert learning_rate(args, *schedule) == pytest.approx(args.learning_rate*.1)


def test_assets_download_one_immutable_revision(monkeypatch):
    import huggingface_hub
    from causal_tuner.checkpointing.huggingface import resolve_assets
    calls = []
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: SimpleNamespace(sha="immutable"))
    def download(**kwargs):
        calls.append(kwargs)
        return "snapshots/immutable"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert resolve_assets("fixture/remote", "main").name == "immutable"
    assert calls[0]["revision"] == "immutable"
    assert "chat_templates/*.jinja" in calls[0]["allow_patterns"]
@pytest.mark.parametrize("extension,delimiter", [("csv", ","), ("tsv", "\t")])
def test_long_csv_fields_and_multiline_records_are_bounded_and_restore_global_limit(tmp_path, extension, delimiter):
    import csv
    from causal_tuner.data.pipeline import local_records
    path = tmp_path/("long."+extension)
    text = "a "*70000 + "\n" + "\u53f0\u5317 "*1000
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["text", "id"])
        writer.writerow([text, "1"])
        writer.writerow(["two three", "2"])
    previous = csv.field_size_limit()
    rows = local_records(path, 2*1024**2)
    assert next(rows) == {"text": text, "id": "1"}
    assert csv.field_size_limit() == previous
    # A second suspended reader must not observe or retain temporary settings.
    other = local_records(path, 2*1024**2)
    assert next(other)["text"] == text
    rows.close()
    assert csv.field_size_limit() == previous
    assert list(other) == [{"text": "two three", "id": "2"}]
    assert csv.field_size_limit() == previous
    with pytest.raises(MemoryError, match="byte budget"):
        list(local_records(path, 1024))
    assert csv.field_size_limit() == previous


@pytest.mark.parametrize("text", ['"'+('one\n'*1000)+'"', '\u53f0'*900, 'one,'*1000])
def test_csv_budget_applies_to_whole_record_and_decoded_memory(tmp_path, text):
    import csv
    from causal_tuner.data.pipeline import local_records
    path = tmp_path/"large.csv"
    path.write_text("text\n"+text+"\n", encoding="utf-8")
    previous = csv.field_size_limit()
    with pytest.raises(MemoryError, match="byte budget"):
        list(local_records(path, 2048))
    assert csv.field_size_limit() == previous

