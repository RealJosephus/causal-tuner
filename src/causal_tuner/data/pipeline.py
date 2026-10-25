"""Bounded raw/prepared iteration, packing, and replayable batch cursors."""

from collections import deque
import csv
from array import array
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
from threading import RLock
import torch
from .segments import canonical_segments, position_ids, prediction_targets, runs
from ..hub import retry, replay

FIELDS = ("input_ids", "attention_mask", "segment_ids", "position_ids", "loss_weights", "labels")
_csv_lock = RLock()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def footprint(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()+256
    if isinstance(value, dict):
        return sys.getsizeof(value)+sum(footprint(k)+footprint(v) for k, v in value.items())
    if isinstance(value, (list, tuple, deque)):
        return sys.getsizeof(value)+sum(map(footprint, value))
    return sys.getsizeof(value)


def compact(record):
    """Own contiguous token buffers; array sizes include their backing storage."""
    return {key: values if key.startswith("_") else array("d" if key.endswith("weights") else "q", values)
            for key, values in record.items()}


def processing_signature(args, tokenizer):
    names = ("max_sequence_length", "dataset_text_field", "packing", "packing_batch_size", "assistant_only_loss",
             "last_assistant_only_loss", "endprompt_enable", "endprompt_logical_length", "endprompt_logical_length_min",
             "endprompt_prompts", "endprompt_prompt_loss_weight", "endprompt_context_loss_weight", "seed")
    return {"version": 1, "tokenizer": tokenizer.signature, **{n: getattr(args, n) for n in names}}


def normalize(record, max_length, vocab):
    ids = record.get("input_ids")
    if not isinstance(ids, (list, tuple)):
        raise ValueError("input_ids must be a one-dimensional list")
    n = len(ids)
    if n == 0:
        return {name: [] for name in FIELDS}
    defaults = {"attention_mask": [1]*n, "segment_ids": [0]*n, "loss_weights": [1.0]*n, "labels": ids}
    result = {name: list(record.get(name, default)) for name, default in {"input_ids": ids, **defaults}.items()}
    if "loss_weights" not in record and "assistant_masks" in record:
        result["loss_weights"] = list(record["assistant_masks"])
    for name in ("position_ids", "attention_end", "prediction_targets", "prediction_weights"):
        if name in record:
            result[name] = list(record[name])
    if ("prediction_targets" in result) != ("prediction_weights" in result):
        raise ValueError("Explicit prediction targets and weights must be supplied together")
    for name, values in result.items():
        if len(values) != n:
            raise ValueError(f"{name} length must match input_ids")
        if name.endswith("weights"):
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("Loss weights must be finite and non-negative")
        elif any(type(x) is not int for x in values):
            raise ValueError(f"{name} must contain integers")
    if any(x not in (0, 1) for x in result["attention_mask"]):
        raise ValueError("attention_mask must be binary")
    if any(not 0 <= x < vocab for x in ids):
        raise ValueError("input_ids contain an invalid vocabulary index")
    for name in ("labels", "prediction_targets"):
        if name in result and any(x != -100 and not 0 <= x < vocab for x in result[name]):
            raise ValueError("Labels contain an invalid vocabulary index")
    result = {k: v[:max_length] for k, v in result.items()}
    mask = torch.tensor([result["attention_mask"]])
    segments = canonical_segments(mask, torch.tensor([result["segment_ids"]]))
    result["segment_ids"] = segments[0].tolist()
    if "attention_end" in result:
        end = result["attention_end"]
        for _, left, right in runs(segments):
            cursor = left
            while cursor < right:
                stop = end[cursor]+1
                if stop < cursor+1 or stop > right or any(x != stop-1 for x in end[cursor:stop]):
                    raise ValueError("Bidirectional groups must be complete intervals inside one segment")
                cursor = stop
    if "position_ids" not in result:
        result["position_ids"] = position_ids(segments)[0].tolist()
    if any(p < 0 for p in result["position_ids"]):
        raise ValueError("Logical positions must be non-negative")
    return result


def supervised(record):
    if len(record["input_ids"]) < 2 and "prediction_targets" not in record:
        return False
    tensors = {k: torch.tensor([v]) for k, v in record.items()}
    return bool((prediction_targets(tensors)[1] > 0).any())


def raw_record(record, tokenizer, args, identity):
    prompt = []
    if args.endprompt_enable:
        prompts = [p for p in args.endprompt_prompts.split("||") if p]
        prompt = tokenizer.encode(prompts[identity % len(prompts)])
        if not prompt or len(prompt) >= args.max_sequence_length:
            raise ValueError("EndPrompt must encode to a nonempty span shorter than the sequence")
    budget = args.max_sequence_length-len(prompt)
    if args.dataset_text_field == "messages":
        ids, weights = tokenizer.messages(record["messages"], assistant_only=args.assistant_only_loss,
                                          last_only=args.last_assistant_only_loss, tools=record.get("tools"),
                                          template_kwargs=record.get("chat_template_kwargs"))
        ids, weights = ids[:budget], weights[:budget]
        if args.assistant_only_loss or args.last_assistant_only_loss:
            indices = [i for i in range(1, len(weights)) if weights[i] > 0]
            if not indices:
                return None
            ids, weights = ids[:indices[-1]+1], weights[:indices[-1]+1]
    else:
        if args.assistant_only_loss or args.last_assistant_only_loss:
            raise ValueError("Assistant loss modes require messages")
        text = record.get(args.dataset_text_field)
        if not isinstance(text, str):
            raise ValueError("Raw text field must contain a string")
        ids = tokenizer.encode(text)[:budget]
        weights = [1.0]*len(ids)
    positions = list(range(len(ids)))
    if prompt:
        width = args.endprompt_logical_length-args.endprompt_logical_length_min+1
        # Stable source identity, independent of shuffle, epochs, and packing.
        logical = args.endprompt_logical_length_min + int(digest([args.seed, identity])[:16], 16) % width
        positions += list(range(logical-len(prompt), logical))
        weights = [w*args.endprompt_context_loss_weight for w in weights]+[args.endprompt_prompt_loss_weight]*len(prompt)
        ids += prompt
    return {"input_ids": ids, "loss_weights": weights, "position_ids": positions}


def bounded_shuffle(records, count, byte_limit, seed):
    rng, pool, used = random.Random(seed), [], 0
    for record in records:
        size = footprint(record)
        if size > byte_limit:
            raise MemoryError(f"One data record requires {size} bytes; stage budget is {byte_limit}")
        while pool and (len(pool) >= count or used+size > byte_limit):
            index = rng.randrange(len(pool))
            value, cost = pool.pop(index)
            used -= cost
            yield value
        pool.append((record, size))
        used += size
    while pool:
        value, _ = pool.pop(rng.randrange(len(pool)))
        yield value


def pad(record, length, pad_id):
    out = {}
    for key, values in record.items():
        if key.startswith("_"):
            out[key] = values
            continue
        fill = pad_id if key == "input_ids" else -100 if key == "labels" else 0
        padding = [fill]*(length-len(values))
        out[key] = values + (array(values.typecode, padding) if isinstance(values, array) else padding)
    return out


def pack_group(group, length, pad_id):
    bins = []
    for record in sorted(group, key=lambda r: -len(r["input_ids"])):
        n = len(record["input_ids"])
        fits = [(length-len(b["input_ids"])-n, index) for index, b in enumerate(bins) if len(b["input_ids"])+n <= length]
        if fits:
            target = bins[min(fits)[1]]
            if target.keys() != record.keys():
                raise ValueError("Packed records must have identical token fields")
            offset = max(target["segment_ids"], default=0)
            physical = len(target["input_ids"])
            for key, values in record.items():
                if key == "_source_count":
                    target[key] += values
                    continue
                if key == "segment_ids":
                    values = [v+offset if v else 0 for v in values]
                elif key == "attention_end":
                    values = [v+physical for v in values]
                target[key].extend(values)
        else:
            bins.append(compact(record))
    return bins


def packing(records, args, pad_id, stage_budget):
    group, used = [], 0
    for record in records:
        size = footprint(record)
        if size > stage_budget//2:
            raise MemoryError(f"Packing one record needs {size*2} bytes including output; stage budget is {stage_budget}")
        if group and (len(group) >= args.packing_batch_size or used+size > stage_budget//2):
            yield from pack_group(group, args.max_sequence_length, pad_id)
            group, used = [], 0
        group.append(record)
        used += size
    if group:
        yield from pack_group(group, args.max_sequence_length, pad_id)


def collate(group, byte_limit):
    """Remove padding and join records without joining attention or label edges."""
    keys = [key for key in group[0] if not key.startswith("_")]
    if any(record.keys() != group[0].keys() for record in group):
        raise ValueError("Batched records must have identical token fields")
    counts = [sum(record["attention_mask"]) for record in group]
    total = sum(counts)
    tensor_bytes = total*sum(4 if key.endswith("weights") else 8 for key in keys)
    # Include original buffers, output tensors, and bounded per-record index /
    # gather temporaries before allocating the materialized batch.
    scratch_bytes = 40*max(len(record["input_ids"]) for record in group)
    if footprint(group)+tensor_bytes+scratch_bytes > byte_limit:
        raise MemoryError("A materialized batch exceeds the data stage budget")
    batch = {key: torch.empty((1, total), dtype=torch.float32 if key.endswith("weights") else torch.long)
             for key in keys}
    physical, segment_offset = 0, 0
    for record, count in zip(group, counts):
        if not count:
            continue
        def tensor(key):
            values = record[key]
            return torch.frombuffer(values, dtype=torch.float64 if key.endswith("weights") else torch.int64)
        mask = tensor("attention_mask")
        indices = mask.nonzero().flatten()
        next_offset = segment_offset+int(tensor("segment_ids")[indices].max())
        for key in keys:
            values = tensor(key)[indices]
            if key == "segment_ids":
                values += segment_offset
            elif key == "attention_end":
                # Bidirectional endpoints refer to physical positions in the
                # original record, including gaps that are now removed.
                values = mask.cumsum(0)[values]-1+physical
            batch[key][0, physical:physical+count].copy_(values)
        physical += count
        segment_offset = next_offset
    batch["_source_count"] = sum(record["_source_count"] for record in group)
    batch["_spans"] = runs(batch["segment_ids"])
    return batch


def jsonl_records(path, byte_limit):
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(byte_limit+1)
            if not line:
                return
            if len(line) > byte_limit:
                raise MemoryError("A JSONL record exceeds the data stage byte budget")
            if line.strip():
                record = json.loads(line)
                if footprint(record) > byte_limit:
                    raise MemoryError("A decoded JSONL record exceeds the data stage byte budget")
                yield record


def local_records(path, byte_limit):
    path = Path(path)
    extension = path.suffix.lower()
    if extension in {".jsonl", ".ndjson"}:
        yield from jsonl_records(path, byte_limit)
    elif extension == ".json":
        with path.open("rb") as handle:
            first = handle.read(1)
            while first and first.isspace():
                first = handle.read(1)
            handle.seek(0)
            if first == b"[":
                import ijson
                yield from ijson.items(handle, "item", use_float=True)
            else:
                yield from jsonl_records(path, byte_limit)
    elif extension in {".csv", ".tsv"}:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            used = 0
            def lines():
                nonlocal used
                while line := handle.readline(byte_limit-used+1):
                    used += len(line.encode("utf-8"))
                    if used > byte_limit:
                        raise MemoryError("A CSV/TSV record exceeds the data stage byte budget")
                    yield line
            reader = csv.DictReader(lines(), delimiter="\t" if extension == ".tsv" else ",")
            # Never hold a process-global setting across a yield: suspended,
            # interleaved and explicitly closed generators leave it unchanged.
            def read(operation):
                nonlocal used
                used = 0
                with _csv_lock:
                    previous = csv.field_size_limit()
                    try:
                        csv.field_size_limit(byte_limit)
                        return operation()
                    finally:
                        csv.field_size_limit(previous)
            fields = read(lambda: reader.fieldnames)
            if footprint(fields) > byte_limit:
                raise MemoryError("A CSV/TSV header exceeds the data stage byte budget")
            while (record := read(lambda: next(reader, None))) is not None:
                if footprint(record) > byte_limit:
                    raise MemoryError("A decoded CSV/TSV record exceeds the data stage byte budget")
                yield record
    elif extension == ".parquet":
        import pyarrow.parquet as pq
        with pq.ParquetFile(path) as handle:
            for batch in handle.iter_batches(batch_size=1):
                yield from batch.to_pylist()
    else:
        raise ValueError("Local datasets require JSON, JSONL, CSV, TSV, or Parquet")


class DataPipeline:
    def __init__(self, args, tokenizer, vocab):
        self.args, self.tokenizer, self.vocab = args, tokenizer, vocab
        self.stage_budget = args.data_buffer_budget_mib*1024**2//4
        self.manifest = None
        self.remote_streaming = False
        self.source_count = None
        self.retry_options = dict(initial_delay=args.hf_retry_initial_delay, max_delay=args.hf_retry_max_delay)
        self.processing = processing_signature(args, tokenizer)
        source = Path(args.dataset_name)
        if args.dataset_format == "prepared":
            from .prepared import load_prepared
            self.dataset, self.manifest, revision = load_prepared(args, self.processing)
            self.remote_streaming = revision is not None and args.dataset_streaming
            self.source_count = self.manifest["records"]
            self.identity = digest([self.manifest, revision])
        elif source.is_file():
            from ..checkpointing.io import file_hash
            self.identity = digest({"sha256": file_hash(source), "processing": self.processing})
        else:
            from datasets import load_dataset
            from huggingface_hub import HfApi
            revision = retry(lambda: HfApi(token=args.token).dataset_info(args.dataset_name, revision=args.dataset_revision).sha,
                             **self.retry_options)
            self.dataset = retry(lambda: load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split,
                                                      revision=revision, token=args.token, streaming=args.dataset_streaming),
                                 **self.retry_options)
            splits = getattr(getattr(self.dataset, "info", None), "splits", None)
            count = getattr(splits.get(args.dataset_split), "num_examples", None) if splits else None
            self.source_count = count if type(count) is int and count > 0 else None
            fingerprint = getattr(self.dataset, "_fingerprint", None)
            self.identity = digest([args.dataset_name, args.dataset_config_name, args.dataset_split,
                                    revision, fingerprint, self.processing])

    def records(self):
        a = self.args
        if self.manifest:
            count = 0
            source = (replay(lambda: iter(self.dataset), fields=self.manifest["features"], **self.retry_options)
                      if self.remote_streaming else iter(self.dataset))
            for record in source:
                count += 1
                if footprint(record) > self.stage_budget:
                    raise MemoryError("A prepared record exceeds the data stage byte budget")
                if (not isinstance(record.get("input_ids"), list)
                        or not 0 < len(record["input_ids"]) <= a.max_sequence_length
                        or any(not isinstance(v, list) or len(v) != len(record["input_ids"]) for v in record.values())):
                    raise ValueError("Prepared records must include aligned token fields within max_sequence_length")
                record = normalize(record, a.max_sequence_length, self.vocab)
                if footprint(record) > self.stage_budget:
                    raise MemoryError("A reconstructed prepared record exceeds the data stage byte budget")
                yield record
            if count != self.manifest["records"]:
                raise ValueError("Prepared dataset record count mismatch")
            return
        # Leave conservative room for Rust tokenization, Python token lists,
        # rendered chat text, and normalization of the same source record.
        raw_limit = self.stage_budget//64
        source = (local_records(a.dataset_name, raw_limit) if Path(a.dataset_name).is_file() else
                  replay(lambda: iter(self.dataset), fields=(a.dataset_text_field, "tools", "chat_template_kwargs"), **self.retry_options))
        for index, record in enumerate(source):
            if footprint(record) > raw_limit:
                raise MemoryError(f"Raw tokenization requires a conservative {footprint(record)*64} byte allowance; stage budget is {self.stage_budget}")
            yield raw_record(record, self.tokenizer, a, index)

    def examples(self, epoch):
        a = self.args
        def normalized():
            pending, previous = 0, None
            for raw in self.records():
                pending += 1
                record = raw
                if raw is not None and not self.manifest:
                    record = normalize(raw, a.max_sequence_length, self.vocab)
                if record is not None and supervised(record):
                    if previous is not None:
                        yield previous
                    record["_source_count"] = pending
                    previous, pending = compact(record), 0
            if previous is not None:
                previous["_source_count"] += pending
                yield previous
        records = normalized()
        if a.shuffle:
            records = bounded_shuffle(records, a.streaming_shuffle_buffer, self.stage_budget, a.seed+epoch)
        if a.packing and self.manifest is None:
            yield from packing(records, a, self.tokenizer.pad_token_id, self.stage_budget)
        else:
            yield from records

    def batches(self, epoch=0, skip=0):
        a = self.args
        iterator = iter(self.examples(epoch))
        index, seen = 0, False
        def generate():
            nonlocal index, seen
            while group := list(itertools.islice(iterator, a.total_batch_size)):
                seen = True
                if index >= skip:
                    yield collate(group, self.stage_budget)
                index += 1
            if not seen:
                raise ValueError("No supervised training data remains after preprocessing")
        queue, used = deque(), 0
        for batch in generate():
            size = footprint(batch)
            if size > self.stage_budget:
                raise MemoryError("A tensor batch exceeds the prefetch byte budget")
            while queue and (len(queue) >= max(1, a.prefetch_batches) or used+size > self.stage_budget):
                value, cost = queue.popleft()
                used -= cost
                yield value
            if not a.prefetch_batches:
                yield batch
            else:
                queue.append((batch, size))
                used += size
        while queue:
            yield queue.popleft()[0]
