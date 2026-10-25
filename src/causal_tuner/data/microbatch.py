"""Repack independent attention spans without changing a logical update."""

import torch
from .segments import canonical_segments, position_ids, runs


def split_batch(batch, token_budget, byte_limit):
    """Yield one pack at a time; an oversized span remains complete."""
    if not token_budget or batch["input_ids"].numel() <= token_budget:
        yield batch
        return
    segments = canonical_segments(batch["attention_mask"], batch.get("segment_ids"))
    spans = batch.get("_spans") or runs(segments)
    if not spans:
        yield batch
        return
    groups, sizes = [], []
    for span in sorted(spans, key=lambda span: span[1]-span[2]):
        length = span[2]-span[1]
        destination = next((i for i, size in enumerate(sizes) if size+length <= token_budget), None)
        if destination is None:
            groups.append([span])
            sizes.append(length)
        else:
            groups[destination].append(span)
            sizes[destination] += length
    if len(groups) == 1 and len(spans) == 1 and spans[0] == (0, 0, batch["input_ids"].numel()):
        yield batch
        return
    fields = {key: value for key, value in batch.items() if not key.startswith("_")}
    if "position_ids" not in fields:
        fields["position_ids"] = position_ids(segments)
    fields["segment_ids"] = segments
    if any(not torch.is_tensor(value) or value.shape != batch["input_ids"].shape for value in fields.values()):
        raise ValueError("Microbatch token fields must match input_ids")
    bytes_per_token = sum(value.element_size() for value in fields.values())
    for index, (group, size) in enumerate(zip(groups, sizes)):
        if 2*size*bytes_per_token > byte_limit:
            raise MemoryError("Repacking a microbatch exceeds the data stage byte budget")
        output = {key: value for key, value in batch.items() if key.startswith("_")}
        output["_source_count"] = batch.get("_source_count", 0) if index == 0 else 0
        offsets, cursor = [], 0
        for _, start, end in group:
            offsets.append(cursor)
            cursor += end-start
        for key, values in fields.items():
            parts = []
            for segment, ((row, start, end), offset) in enumerate(zip(group, offsets), 1):
                part = values[row, start:end]
                if key == "segment_ids":
                    part = torch.full_like(part, segment)
                elif key == "attention_end":
                    part = part-start+offset
                parts.append(part)
            output[key] = torch.cat(parts).unsqueeze(0)
        output["_spans"] = tuple((0, offset, offset+end-start)
                                 for (_, start, end), offset in zip(group, offsets))
        yield output


def _pack_signature(batch):
    """Only join complete host packs with known positional metadata."""
    ids = batch["input_ids"]
    spans = batch.get("_spans", ())
    if ids.device.type != "cpu" or ids.ndim != 2 or ids.shape[0] != 1 or not spans:
        return None
    cursor = 0
    for row, start, end in spans:
        if row != 0 or start != cursor or end <= start:
            return None
        cursor = end
    if cursor != ids.numel() or any(key.startswith("_") and key not in {"_spans", "_source_count"} for key in batch):
        return None
    fields = []
    for key, value in sorted(batch.items()):
        if key.startswith("_"):
            continue
        if not torch.is_tensor(value) or value.device.type != "cpu" or value.shape != ids.shape:
            return None
        fields.append((key, value.dtype))
    return tuple(fields)


def _join_packs(parts):
    if len(parts) == 1:
        return parts[0]
    offsets, spans, rows = [], [], 0
    for part in parts:
        offsets.append(rows)
        spans.extend((0, rows+start, rows+end) for _, start, end in part["_spans"])
        rows += part["input_ids"].numel()
    output = {}
    for key, value in parts[0].items():
        if key.startswith("_") or key == "segment_ids":
            continue
        values = [part[key]+offset if key == "attention_end" else part[key]
                  for part, offset in zip(parts, offsets)]
        output[key] = torch.cat(values, dim=1)
    output["segment_ids"] = torch.empty((1, rows), dtype=parts[0].get("segment_ids", parts[0]["input_ids"]).dtype)
    for segment, (_, start, end) in enumerate(spans, 1):
        output["segment_ids"][0, start:end] = segment
    output["_spans"] = tuple(spans)
    output["_source_count"] = sum(part.get("_source_count", 0) for part in parts)
    return output


def repack_batches(batches, token_budget, byte_limit):
    """Coalesce compatible short packs inside one optimizer update."""
    if not token_budget:
        yield from batches
        return
    from .pipeline import footprint
    pending, rows, signature = [], 0, None
    for batch in batches:
        for part in split_batch(batch, token_budget, byte_limit):
            size = part["input_ids"].numel()
            current = _pack_signature(part) if size < token_budget else None
            fits = pending and current is not None and current == signature and rows+size <= token_budget
            if fits:
                # Account for held inputs, the joined output, and field-copy
                # temporaries before retaining another input batch.
                candidates = pending+[part]
                tensor_bytes = sum(value.numel()*value.element_size() for piece in candidates
                                   for key, value in piece.items() if not key.startswith("_"))
                fits = footprint(candidates)+2*tensor_bytes <= byte_limit
            if pending and not fits:
                yield _join_packs(pending)
                pending, rows = [], 0
            if current is None:
                yield part
                continue
            pending.append(part)
            rows += size
            signature = current
            if rows == token_budget:
                yield _join_packs(pending)
                pending, rows = [], 0
    if pending:
        yield _join_packs(pending)
