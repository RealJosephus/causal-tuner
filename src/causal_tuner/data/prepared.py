"""Hugging Face Datasets storage with preprocessing provenance."""

from contextlib import contextmanager
import json
from pathlib import Path
from threading import RLock

from ..hub import retry

_saving_lock = RLock()


def features_for(columns):
    from datasets import Features, Sequence, Value
    supported = {"input_ids", "attention_mask", "segment_ids", "position_ids", "loss_weights", "labels",
                 "attention_end", "prediction_targets", "prediction_weights"}
    if not set(columns) <= supported:
        raise ValueError("Unsupported prepared token fields")
    return Features({name: Sequence(Value("float64" if name.endswith("weights") else
                                          "int8" if name == "attention_mask" else "int32"))
                     for name in columns})


def default_fields(record, candidates):
    """Identify token fields that normalization can reconstruct exactly."""
    result = set()
    for name, value in (("attention_mask", 1), ("segment_ids", 1), ("loss_weights", 1.0)):
        if name in candidates and all(item == value for item in record[name]):
            result.add(name)
    if "labels" in candidates and record["labels"] == record["input_ids"]:
        result.add("labels")
    if "position_ids" in candidates:
        previous, start = 0, 0
        for index, (segment, position) in enumerate(zip(record["segment_ids"], record["position_ids"])):
            if segment > 0 and segment != previous:
                start = index
            if position != (index-start if segment > 0 else 0):
                break
            previous = segment
        else:
            result.add("position_ids")
    return result


@contextmanager
def saving_batch_size(rows, *, dataset=None):
    """Bound native Arrow saves and the Parquet batches selected by push_to_hub."""
    from datasets import config
    from datasets.utils.py_utils import convert_file_size_to_int
    if rows < 1:
        raise ValueError("Serialization batch size must be positive")
    with _saving_lock:
        previous = config.DEFAULT_MAX_BATCH_SIZE, config.MAX_ROW_GROUP_SIZE
        try:
            rows = min(previous[0], rows)
            config.DEFAULT_MAX_BATCH_SIZE = rows
            if dataset is not None:
                # Prepared features contain only numeric lists. Datasets selects
                # floor(num_rows * MAX_ROW_GROUP_SIZE / estimated_bytes) rows,
                # then passes that count explicitly to to_parquet. Invert that
                # formula so even skewed lengths respect the largest-row bound.
                size = dataset._estimate_nbytes()
                limit = max(1, rows*size//max(1, len(dataset)))
                config.MAX_ROW_GROUP_SIZE = min(convert_file_size_to_int(previous[1]), limit)
            yield
        finally:
            config.DEFAULT_MAX_BATCH_SIZE, config.MAX_ROW_GROUP_SIZE = previous


def load_prepared(args, processing):
    from datasets import Dataset, Features, load_dataset, load_from_disk
    from ..checkpointing.io import file_hash, safe_path
    source = Path(args.dataset_name)
    options = dict(initial_delay=args.hf_retry_initial_delay, max_delay=args.hf_retry_max_delay)
    revision = None
    if source.is_dir():
        manifest_path = source/"manifest.json"
    else:
        from huggingface_hub import HfApi, hf_hub_download
        revision = retry(lambda: HfApi(token=args.token).dataset_info(args.dataset_name,
                         revision=args.dataset_revision).sha, **options)
        manifest_path = Path(retry(lambda: hf_hub_download(args.dataset_name, "manifest.json", repo_type="dataset",
                                  revision=revision, token=args.token), **options))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") not in (2, 3) or manifest.get("format") != "huggingface_datasets":
        raise ValueError("Unsupported prepared format; rebuild the dataset with causal-tune-prepare")
    if manifest.get("processing") != processing:
        raise ValueError("Prepared tokenizer or processing signature mismatch")
    if type(manifest.get("records")) is not int or manifest["records"] <= 0:
        raise ValueError("Prepared manifest record count is invalid")
    if source.is_dir():
        for name, checksum in manifest["files"].items():
            if file_hash(safe_path(source, name)) != checksum:
                raise ValueError("Prepared dataset checksum is invalid")
        dataset = load_from_disk(str(source), keep_in_memory=False)
        if not isinstance(dataset, Dataset):
            raise ValueError("Prepared input must be one Dataset, not a DatasetDict")
    else:
        dataset = retry(lambda: load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split,
                          revision=revision, token=args.token, streaming=args.dataset_streaming), **options)
    if dataset.features != Features.from_dict(manifest["features"]):
        raise ValueError("Prepared dataset features differ from its processing manifest")
    if isinstance(dataset, Dataset) and len(dataset) != manifest["records"]:
        raise ValueError("Prepared dataset record count mismatch")
    return dataset, manifest, revision
