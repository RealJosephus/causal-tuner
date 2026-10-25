"""Native Datasets interchange, compact storage, and deterministic replay."""

import copy
import json
from pathlib import Path

import pytest
import torch

from causal_tuner.data.pipeline import DataPipeline, normalize
from causal_tuner.prepare import prepare_dataset


def source_args(tmp_path, args, rows):
    source = tmp_path/"source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    args.dataset_name, args.output_dir = str(source), str(tmp_path/"prepared")
    return args


def prepared_args(args):
    result = copy.copy(args)
    result.dataset_format, result.dataset_name = "prepared", args.output_dir
    return result


def equal_batches(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for key in a:
            if torch.is_tensor(a[key]):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            else:
                assert a[key] == b[key]


@pytest.mark.parametrize("packing,version", [(False, 2), (True, 3)])
@pytest.mark.parametrize("epochs,batch_size,accumulation", [(1., 1, 1), (.5, 2, 3), (2.25, 2, 2)])
def test_prepared_epoch_plan_uses_metadata_without_reading_tokens(tmp_path, args, tokenizer, monkeypatch,
                                                               packing, version, epochs, batch_size, accumulation):
    import math
    from causal_tuner.train import training_plan
    source_args(tmp_path, args, [{"text": text} for text in
                                ("one", "one two", "", "one two three", "three two", "one three", "two three")])
    args.packing = packing
    directory = prepare_dataset(args, tokenizer, 97)
    manifest_path = directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = version
    manifest_path.write_text(json.dumps(manifest))
    current = prepared_args(args)
    current.shuffle, current.streaming_shuffle_buffer = True, 2
    current.num_train_epochs, current.total_batch_size = epochs, batch_size
    current.gradient_accumulation_steps = accumulation
    pipeline = DataPipeline(current, tokenizer, 97)
    expected = [math.ceil(sum(1 for _ in pipeline.batches(epoch))*min(1., epochs-epoch))
                for epoch in range(math.ceil(epochs))]
    def forbidden(*a, **kw):
        pytest.fail("Planning prepared epochs must not read token batches")
    monkeypatch.setattr(pipeline, "batches", forbidden)
    assert training_plan(current, pipeline) == {
        "steps": sum(math.ceil(n/accumulation) for n in expected),
        "batches": expected, "source_per_epoch": None}


def test_native_dataset_roundtrip_preserves_ragged_values_and_precision(tmp_path, args, tokenizer):
    from datasets import Dataset, load_dataset, load_from_disk
    rows = [{"messages": [{"role": "user", "content": "one"},
                          {"role": "assistant", "content": text, "weight": .12345678901234567}]}
            for text in ("one two", "two three one", "three one two three")]
    source_args(tmp_path, args, rows)
    args.dataset_text_field, args.assistant_only_loss = "messages", True
    args.max_sequence_length = 131072
    expected = [{k: list(v) for k, v in row.items() if not k.startswith("_")}
                for row in DataPipeline(args, tokenizer, 97).examples(0)]
    directory = prepare_dataset(args, tokenizer, 97)
    dataset = load_from_disk(str(directory))
    assert isinstance(dataset, Dataset)
    stored = list(dataset)
    assert [normalize(row, args.max_sequence_length, 97) for row in stored] == expected
    assert set(dataset.column_names) == {"input_ids", "loss_weights"}
    assert dataset.features["input_ids"].feature.dtype == "int32"
    assert dataset.features["loss_weights"].feature.dtype == "float64"
    assert len({len(row["input_ids"]) for row in dataset}) == 3
    assert (directory/"state.json").is_file() and (directory/"dataset_info.json").is_file()
    assert list(directory.glob("*.arrow")) and not list(directory.glob("*.jsonl"))
    assert sum(p.stat().st_size for p in directory.iterdir()) < 32*1024
    # The library's Parquet path is also used by Dataset.push_to_hub. Exercise
    # the actual conversion and loader without creating a remote repository.
    parquet = tmp_path/"train.parquet"
    dataset.to_parquet(str(parquet))
    downloaded = load_dataset("parquet", data_files=str(parquet), split="train", cache_dir=str(tmp_path/"hf-cache"))
    assert downloaded.features == dataset.features
    assert list(downloaded) == stored
    streamed = load_dataset("parquet", data_files=str(parquet), split="train", streaming=True)
    assert streamed.features == dataset.features
    assert list(streamed) == stored


def test_last_assistant_packing_stores_only_needed_columns(tmp_path, args, tokenizer):
    from datasets import load_from_disk
    rows = [{"messages": [{"role": "user", "content": "one two"},
                          {"role": "assistant", "content": "three"},
                          {"role": "user", "content": "two"},
                          {"role": "assistant", "content": answer}]}
            for answer in ("one", "two three", "one two three")]
    source_args(tmp_path, args, rows)
    args.dataset_text_field, args.last_assistant_only_loss = "messages", True
    args.max_sequence_length, args.packing = 131072, True
    expected = [{key: list(value) for key, value in row.items() if not key.startswith("_")}
                for row in DataPipeline(args, tokenizer, 97).examples(0)]
    directory = prepare_dataset(args, tokenizer, 97)
    dataset = load_from_disk(str(directory))
    assert set(dataset.column_names) == {"input_ids", "segment_ids", "loss_weights"}
    assert list(DataPipeline(prepared_args(args), tokenizer, 97).records()) == expected


def test_native_list_columns_use_library_compression(tmp_path):
    datasets = pytest.importorskip("datasets", minversion="5.1.0")
    import pyarrow.parquet as pq
    from causal_tuner.data.prepared import features_for
    size = 8192
    values = {"input_ids": [[index % 97 for index in range(size)] for _ in range(2)],
              "attention_mask": [[1]*size]*2, "loss_weights": [[0.]*(size-128)+[1.]*128]*2}
    dataset = datasets.Dataset.from_dict(values, features=features_for(values))
    path = tmp_path/"native.parquet"
    dataset.to_parquet(path)
    metadata = pq.ParquetFile(path).metadata
    assert all(metadata.row_group(0).column(index).compression != "UNCOMPRESSED"
               for index in range(metadata.num_columns))
    assert path.stat().st_size < dataset.data.nbytes//4


def test_storage_size_tracks_actual_tokens_instead_of_the_length_cap(tmp_path, args, tokenizer):
    from datasets import load_from_disk
    source_args(tmp_path, args, [{"text": "one two three "*7} for _ in range(12)])
    sizes, values = [], []
    for cap in (64, 131072):
        args.max_sequence_length, args.output_dir = cap, str(tmp_path/f"prepared-{cap}")
        directory = prepare_dataset(args, tokenizer, 97)
        sizes.append(sum(p.stat().st_size for p in directory.iterdir()))
        values.append(list(load_from_disk(str(directory))))
    assert values[0] == values[1]
    assert abs(sizes[0]-sizes[1]) < 1024


@pytest.mark.parametrize("packing", [False, True])
def test_native_dataset_replay_without_tokenizing_twice(tmp_path, args, tokenizer, monkeypatch, packing):
    from datasets import load_from_disk
    source_args(tmp_path, args, [{"text": text} for text in ("one two", "two three one", "one three", "three two")])
    args.packing, args.endprompt_enable = packing, True
    args.endprompt_prompts = "anchor"
    args.endprompt_logical_length, args.endprompt_logical_length_min = 2000000, 1999990
    args.total_batch_size = 2
    expected = list(DataPipeline(args, tokenizer, 97).batches())
    directory = prepare_dataset(args, tokenizer, 97)
    assert max(p for row in load_from_disk(str(directory)) for p in row["position_ids"]) > 1900000
    monkeypatch.setattr(tokenizer, "encode", lambda *a, **k: pytest.fail("Prepared input must not tokenize"))
    current = prepared_args(args)
    actual = list(DataPipeline(current, tokenizer, 97).batches())
    # Packing counts stored rows as its source units, as in the former format.
    for a, b in zip(expected, actual):
        a["_source_count"] = b["_source_count"]
    equal_batches(expected, actual)
    equal_batches(actual[1:], list(DataPipeline(current, tokenizer, 97).batches(skip=1)))
    current.shuffle = True
    current.streaming_shuffle_buffer = 2
    for epoch in (0, 1, 3):
        all_batches = list(DataPipeline(current, tokenizer, 97).batches(epoch))
        equal_batches(all_batches[1:], list(DataPipeline(current, tokenizer, 97).batches(epoch, skip=1)))


@pytest.mark.parametrize("version", [2, 3])
def test_optional_fields_gaps_and_large_positions_survive_native_storage(tmp_path, args, tokenizer, monkeypatch, version):
    from datasets import load_from_disk
    source_args(tmp_path, args, [{"text": "one two"}])
    rows = [dict(input_ids=[2, 3, 4, 5, 6], attention_mask=[1, 1, 0, 1, 1],
                 position_ids=[4194302, 4194303, 0, 5, 6], segment_ids=[0]*5,
                 labels=[2, -100, -100, 5, 6], loss_weights=[0., .2, 0., 0., .7],
                 prediction_targets=[3, 4, -100, 6, 7], prediction_weights=[.3, .2, 0., .1, .7],
                 attention_end=[1, 1, 2, 4, 4]),
            dict(input_ids=[8, 9, 10], labels=[8, -100, 10], loss_weights=[0., .5, 1.],
                 prediction_targets=[9, -100, 8], prediction_weights=[.2, 0., .8], attention_end=[0, 2, 2])]
    normalized = [normalize(row, args.max_sequence_length, 97) for row in rows]
    original = DataPipeline.records
    def records(self):
        if self.args.dataset_format == "raw":
            yield from normalized
        else:
            yield from original(self)
    monkeypatch.setattr(DataPipeline, "records", records)
    before = list(DataPipeline(args, tokenizer, 97).batches())
    directory = prepare_dataset(args, tokenizer, 97)
    assert list(load_from_disk(str(directory))) == normalized
    manifest_path = directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = version
    manifest_path.write_text(json.dumps(manifest))
    equal_batches(before, list(DataPipeline(prepared_args(args), tokenizer, 97).batches()))


def test_defaults_are_removed_only_when_all_records_match(tmp_path, args, tokenizer, monkeypatch):
    from datasets import load_from_disk
    source_args(tmp_path, args, [{"text": "one two"}])
    rows = [normalize({"input_ids": [2, 3, 4]}, args.max_sequence_length, 97),
            normalize({"input_ids": [5, 6, 7, 8], "attention_mask": [1, 0, 1, 1],
                       "labels": [5, -100, 7, 8], "position_ids": [30, 0, 40, 41]}, args.max_sequence_length, 97)]
    original = DataPipeline.records
    def records(self):
        if self.args.dataset_format == "raw":
            yield from rows
        else:
            yield from original(self)
    monkeypatch.setattr(DataPipeline, "records", records)
    directory = prepare_dataset(args, tokenizer, 97)
    dataset = load_from_disk(str(directory))
    assert {"attention_mask", "labels", "position_ids", "segment_ids"} <= set(dataset.column_names)
    assert "loss_weights" not in dataset.column_names
    assert list(DataPipeline(prepared_args(args), tokenizer, 97).records()) == rows


def test_prepared_dataset_checkpoint_continuation_is_exact(tmp_path, args, tokenizer, tiny_config):
    from causal_tuner.checkpointing.io import latest_checkpoint, restore_checkpoint, save_checkpoint
    from causal_tuner.modeling.model import CausalLM
    from causal_tuner.train import runtime_from_args
    from causal_tuner.training.engine import Trainer, make_optimizer
    source_args(tmp_path, args, [{"text": text} for text in ("one two three", "three two", "two three one", "one three")])
    prepare_dataset(args, tokenizer, 97)
    current = prepared_args(args)
    current.shuffle, current.streaming_shuffle_buffer = True, 2
    initial = CausalLM(tiny_config, runtime_from_args(current))
    uninterrupted = copy.deepcopy(initial)
    optimizer = make_optimizer(uninterrupted, current)
    trainer = Trainer(uninterrupted, optimizer, current, device="cpu")
    pipeline = DataPipeline(current, tokenizer, 97)
    batches = list(pipeline.batches())
    trainer.update(batches[:2], step=0, total_steps=2)
    signature = {"data": pipeline.identity}
    save_checkpoint(tmp_path/"checkpoints", uninterrupted, optimizer, step=1, cursor={"epoch": 0, "batch": 2},
                    signature=signature, scheduler={"step": 1, "total": 2}, save_optimizer=True)
    trainer.update(batches[2:], step=1, total_steps=2)
    restored_pipeline = DataPipeline(current, tokenizer, 97)
    assert restored_pipeline.identity == pipeline.identity
    restored_optimizer = make_optimizer(initial, current)
    step, cursor = restore_checkpoint(*latest_checkpoint(tmp_path/"checkpoints", signature), initial, restored_optimizer)
    Trainer(initial, restored_optimizer, current, device="cpu").update(
        restored_pipeline.batches(cursor["epoch"], skip=cursor["batch"]), step=step, total_steps=2)
    for p, q in zip(uninterrupted.parameters(), initial.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        if p in optimizer.state:
            for key, value in optimizer.state[p].items():
                torch.testing.assert_close(value, restored_optimizer.state[q][key], rtol=0, atol=0)


def test_failed_native_save_is_not_published_and_restores_library_settings(tmp_path, args, tokenizer, monkeypatch):
    from datasets import Dataset, config
    source_args(tmp_path, args, [{"text": "one two"}])
    previous = config.DEFAULT_MAX_BATCH_SIZE
    args.shuffle = True
    def failed(self, path, **kwargs):
        Path(path).mkdir()
        (Path(path)/"partial.arrow").write_bytes(b"partial")
        raise OSError("failed save")
    monkeypatch.setattr(Dataset, "save_to_disk", failed)
    with pytest.raises(OSError, match="failed save"):
        prepare_dataset(args, tokenizer, 97)
    assert not Path(args.output_dir).exists()
    assert not list(tmp_path.glob(".preparing-*"))
    assert config.DEFAULT_MAX_BATCH_SIZE == previous and args.shuffle


def test_prepared_old_format_and_changed_signature_are_explicit_errors(tmp_path, args, tokenizer):
    source_args(tmp_path, args, [{"text": "one two"}])
    directory = prepare_dataset(args, tokenizer, 97)
    current = prepared_args(args)
    current.seed += 1
    with pytest.raises(ValueError, match="signature"):
        DataPipeline(current, tokenizer, 97)
    current.seed -= 1
    (directory/"manifest.json").write_text(json.dumps({"version": 1, "processing": {}}))
    with pytest.raises(ValueError, match="rebuild"):
        DataPipeline(current, tokenizer, 97)


@pytest.mark.parametrize("changed", ["records", "features"])
def test_native_data_must_match_preparation_metadata(tmp_path, args, tokenizer, changed):
    source_args(tmp_path, args, [{"text": "one two"}])
    directory = prepare_dataset(args, tokenizer, 97)
    manifest_path = directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if changed == "records":
        manifest["records"] += 1
    else:
        manifest["features"]["input_ids"]["feature"]["dtype"] = "int64"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="record count|features"):
        DataPipeline(prepared_args(args), tokenizer, 97)
