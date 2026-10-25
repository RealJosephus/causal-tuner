"""All consumers share contiguous valid runs, including valid raw label zero."""

import torch


def canonical_segments(mask, labels=None):
    valid = mask.bool()
    if valid.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    labels = torch.zeros_like(mask) if labels is None else labels
    if labels.shape != mask.shape:
        raise ValueError("segment_ids must match attention_mask")
    previous = torch.nn.functional.pad(valid[:, :-1], (1, 0), value=False)
    changed = torch.nn.functional.pad(labels[:, 1:] != labels[:, :-1], (1, 0), value=True)
    starts = valid & (~previous | changed)
    return starts.to(torch.int32).cumsum(1).to(torch.int32) * valid


def position_ids(segments):
    index = torch.arange(segments.shape[1], device=segments.device).expand_as(segments)
    previous = torch.nn.functional.pad(segments[:, :-1], (1, 0))
    starts = torch.where((segments != previous) & (segments > 0), index, 0)
    return torch.where(segments > 0, index - starts.cummax(1).values, 0)


def runs(segments):
    """Compact host metadata, made once per batch and shared across layers."""
    result = []
    for row, labels in enumerate(segments.cpu().tolist()):
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and labels[end] == labels[start]:
                end += 1
            if labels[start] > 0:
                result.append((row, start, end))
            start = end
    return tuple(result)


def prediction_targets(batch):
    ids = batch["input_ids"]
    valid = batch["attention_mask"].bool()
    if "prediction_targets" in batch:
        targets = batch["prediction_targets"]
        weights = batch["prediction_weights"].float() * valid
    else:
        segments = canonical_segments(valid, batch.get("segment_ids"))
        labels = batch.get("labels", ids)
        weights = batch.get("loss_weights")
        weights = torch.ones_like(ids, dtype=torch.float32) if weights is None else weights.float()
        if "loss_weights" not in batch and "assistant_masks" in batch:
            weights = weights * batch["assistant_masks"]
        edges = valid[:, :-1] & valid[:, 1:] & (segments[:, :-1] == segments[:, 1:])
        targets = torch.nn.functional.pad(labels[:, 1:], (0, 1))
        weights = torch.nn.functional.pad(weights[:, 1:] * edges, (0, 1))
    weights = weights * (targets != -100)
    return torch.where(weights > 0, targets, 0).long(), weights
