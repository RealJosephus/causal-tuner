"""Prepare a standard Hugging Face Dataset without padding stored records."""

import copy
import gc
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from .arguments import parse_args


def prepare_dataset(args, tokenizer, vocab):
    from datasets import Dataset
    from .data.pipeline import DataPipeline, digest, footprint
    from .data.prepared import default_fields, features_for, saving_batch_size
    from .checkpointing.io import json_write, file_hash
    if args.dataset_format != "raw":
        raise ValueError("Preparation requires raw input; prepared input must not be transformed twice")
    repository = None
    if args.hub_dataset_repo_id:
        from .publishing import HubRepository
        repository = HubRepository(args.hub_dataset_repo_id, "dataset", args)
    destination = Path(args.output_dir)
    if repository is None:
        if destination.exists():
            raise FileExistsError("Prepared output directory already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
    args = copy.copy(args)
    args.shuffle = False
    pipeline = DataPipeline(args, tokenizer, vocab)
    records = iter(pipeline.examples(0))
    first = next(records, None)
    if first is None:
        raise ValueError("No supervised data to prepare")
    columns = [key for key in first if not key.startswith("_")]
    features = features_for(columns)
    largest = 0
    redundant = {"attention_mask", "segment_ids", "loss_weights", "labels", "position_ids"} & set(columns)

    def generate():
        nonlocal largest
        for record in itertools.chain((first,), records):
            if set(key for key in record if not key.startswith("_")) != set(columns):
                raise ValueError("Prepared records must have identical token fields")
            redundant.intersection_update(default_fields(record, redundant))
            largest = max(largest, footprint(record))
            yield {key: list(record[key]) for key in columns}

    # Buffer only a bounded number of Python rows before the library writes
    # them. The final Arrow save uses the largest actual row, not the cap.
    writer_rows = max(1, min(1000, pipeline.stage_budget//(args.max_sequence_length*len(columns)*80)))
    fingerprint = digest([pipeline.identity, args.data_buffer_budget_mib, features.to_dict()])
    temporary = destination.with_name(".preparing-"+uuid.uuid4().hex)
    try:
        with tempfile.TemporaryDirectory(prefix="causal-tuner-prepare-") as cache:
            dataset = None
            try:
                dataset = Dataset.from_generator(generate, features=features, cache_dir=cache,
                                                 fingerprint=fingerprint, writer_batch_size=writer_rows)
                if redundant:
                    dataset = dataset.remove_columns(sorted(redundant))
                features = dataset.features
                if repository is not None:
                    from .publishing import json_bytes
                    with saving_batch_size(max(1, pipeline.stage_budget//(4*largest)), dataset=dataset):
                        uploaded = repository.call(lambda: dataset.push_to_hub(
                            repository.repo_id, config_name="default", split="train", token=args.token,
                            commit_message="Upload prepared training data"))
                    manifest = {"version": 3, "format": "huggingface_datasets", "processing": pipeline.processing,
                                "source": pipeline.identity, "records": len(dataset), "features": features.to_dict(),
                                "files": {}}
                    repository.commit([], parent=uploaded.oid, marker="manifest.json", content=json_bytes(manifest),
                                      message="Publish complete prepared training data")
                    return repository.url()
                with saving_batch_size(max(1, pipeline.stage_budget//(4*largest))):
                    dataset.save_to_disk(str(temporary))
                total = len(dataset)
            finally:
                dataset = None
                # Release memory maps before removing the build cache on Windows.
                gc.collect()
        files = {}
        for path in sorted(temporary.iterdir()):
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
            files[path.name] = file_hash(path)
        json_write(temporary/"manifest.json", {"version": 3, "format": "huggingface_datasets",
                   "processing": pipeline.processing, "source": pipeline.identity, "records": total,
                   "features": features.to_dict(), "files": files})
        sync_directory(temporary)
        temporary.replace(destination)
        sync_directory(destination.parent)
    finally:
        records.close()
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main(argv=None):
    args = parse_args(argv, prepare=True)
    from .checkpointing.huggingface import resolve_assets
    from .data.tokenizer import TrainingTokenizer
    from .modeling.config import ModelConfig
    directory = resolve_assets(args.repo_id, args.revision, args.token, weights=False,
                                initial_delay=args.hf_retry_initial_delay, max_delay=args.hf_retry_max_delay)
    processor = (resolve_assets(args.processor_repo_id or args.repo_id, args.processor_revision,
                                args.token, weights=False, initial_delay=args.hf_retry_initial_delay,
                                max_delay=args.hf_retry_max_delay)
                 if args.processor_repo_id or args.processor_revision else directory)
    config = ModelConfig.from_dict(json.loads((directory/"config.json").read_text()))
    print(prepare_dataset(args, TrainingTokenizer(processor), config.vocab_size))


if __name__ == "__main__":
    main()
