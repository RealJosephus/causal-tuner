"""Execution choices and update-preserving microbatch reconstruction."""

import copy
from dataclasses import replace
import pytest
import torch

from causal_tuner.data.microbatch import repack_batches, split_batch
from causal_tuner.data.segments import canonical_segments, position_ids, prediction_targets, runs
from causal_tuner.modeling.config import ModelConfig
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.training.engine import Trainer, make_optimizer
from causal_tuner.training.execution import execution_policy, GIB


@pytest.mark.parametrize("rows,checkpoint,cache,tile", [
    (8192, False, 28, 0), (12288, False, 28, 0), (16384, False, 24, 0),
    (32768, False, 0, 4096), (131072, True, 0, 4096)])
def test_calibrated_policy_uses_actual_rows(rows, checkpoint, cache, tile):
    plan = execution_policy(ModelConfig(), Runtime(adaptive=True), rows, rows,
                            cuda=True, offload=True, quantized=False)
    assert plan.calibrated and plan.checkpoint == checkpoint
    assert plan.mlp_cache_bytes == cache*GIB and plan.mlp_tile == tile
    assert plan.attention_cache_bytes == (38*GIB if rows == 131072 else 0)


def test_automatic_cache_leaves_room_for_other_activations_and_adam():
    runtime = Runtime(adaptive=True)
    def choose(rows, available):
        return execution_policy(ModelConfig(), runtime, rows, rows, cuda=True, offload=True,
                                quantized=False, available_bytes=available)
    assert choose(8192, 60*GIB).mlp_cache_bytes == 28*GIB
    constrained = choose(15360, 60*GIB)
    assert not constrained.checkpoint
    assert 0 < constrained.mlp_cache_bytes < 24*GIB
    smaller = choose(15360, 50*GIB)
    assert smaller.mlp_cache_bytes < constrained.mlp_cache_bytes
    checkpointed = choose(12288, 24*GIB)
    assert checkpointed.checkpoint and checkpointed.mlp_tile == 4096
    assert checkpointed.mlp_cache_bytes == 0
    long = choose(131072, 35*GIB)
    assert long.checkpoint and 0 < long.attention_cache_bytes < 38*GIB
    assert choose(131072, 0).attention_cache_bytes == 0


def test_policy_respects_explicit_controls_and_unsupported_configurations():
    runtime = Runtime(adaptive=True)
    def choose(r=runtime, **extra):
        return execution_policy(ModelConfig(), r, 131072, 8192,
                                **dict(cuda=True, offload=True, quantized=False, **extra))
    assert choose().attention_cache_bytes == 0
    assert not choose(replace(runtime, rank=128)).calibrated
    assert not choose(replace(runtime, train_head=True)).calibrated
    assert not choose(replace(runtime, group_size=2)).calibrated
    assert not execution_policy(ModelConfig(), runtime, 8192, 8192,
                                cuda=True, offload=True, quantized=True).calibrated
    plan = choose(replace(runtime, mlp_tile=7, attention_cache_budget_bytes=GIB))
    assert plan.mlp_tile == 7 and plan.attention_cache_bytes == GIB
    for rows in (65536, 131072):
        assert execution_policy(ModelConfig(), runtime, rows, rows, cuda=True, offload=False,
                                quantized=False).attention_cache_bytes == 0
    for rows, grouped in ((262144, False), (131072, True)):
        assert execution_policy(ModelConfig(), runtime, rows, rows, cuda=True, offload=True,
                                quantized=False, grouped_attention=grouped).attention_cache_bytes == 0


def sample(explicit):
    ids = torch.arange(2, 20).reshape(2, 9)
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 1, 1, 1, 1, 1]])
    segments = canonical_segments(mask)
    batch = dict(input_ids=ids, attention_mask=mask, segment_ids=torch.zeros_like(ids),
                 position_ids=position_ids(segments)+100, labels=ids.clone(),
                 loss_weights=torch.arange(18).reshape(2, 9).float()/10, _source_count=7)
    batch["labels"][0, 4] = -100
    if explicit:
        batch["prediction_targets"] = ids.flip(-1)
        batch["prediction_weights"] = mask.float()/3
        ends = torch.zeros_like(ids)
        for row, start, stop in runs(segments):
            ends[row, start:stop] = stop-1
        batch["attention_end"] = ends
    return batch


@pytest.mark.parametrize("explicit", [False, True])
def test_repacking_preserves_every_token_target_position_and_attention_group(explicit):
    original = sample(explicit)
    snapshots = {k: v.clone() for k, v in original.items() if torch.is_tensor(v)}
    expected_targets, expected_weights = prediction_targets(original)
    lookup = {int(original['input_ids'][row, index]): (row, index)
              for row, start, end in runs(canonical_segments(original['attention_mask']))
              for index in range(start, end)}
    seen = []
    pieces = list(split_batch(original, 4, 1024**2))
    assert sorted(piece['input_ids'].numel() for piece in pieces) == [2, 3, 5, 5]
    assert sum(piece['_source_count'] for piece in pieces) == 7
    for piece in pieces:
        targets, weights = prediction_targets(piece)
        for i, token in enumerate(piece['input_ids'][0].tolist()):
            row, index = lookup[token]
            seen.append(token)
            assert piece['position_ids'][0, i] == original['position_ids'][row, index]
            assert targets[0, i] == expected_targets[row, index]
            assert weights[0, i] == expected_weights[row, index]
        for _, left, right in piece['_spans']:
            if explicit:
                assert torch.all(piece['attention_end'][0, left:right] == right-1)
    assert sorted(seen) == sorted(lookup)
    for key, value in snapshots.items():
        torch.testing.assert_close(original[key], value, rtol=0, atol=0)
    assert next(split_batch(original, 0, 1)) is original
    with pytest.raises(MemoryError, match='Repacking'):
        list(split_batch(original, 4, 1))


@pytest.mark.parametrize("explicit", [False, True])
def test_repacked_update_matches_weighted_unsplit_update(args, tiny_config, explicit):
    args.warmup_steps = 0
    model = CausalLM(tiny_config, Runtime(rank=4, base_dtype=torch.float32, compute_dtype=torch.float32,
                                        attention='vanilla', train_embed=True, train_head=True))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.01)
    control = copy.deepcopy(model)
    data = sample(explicit)
    options = copy.copy(args)
    args.microbatch_token_budget, options.microbatch_token_budget = 4, 0
    actual = Trainer(model, make_optimizer(model, args), args, device='cpu').update([data], step=0, total_steps=2)
    expected = Trainer(control, make_optimizer(control, options), options, device='cpu').update([data], step=0, total_steps=2)
    assert actual['input_batches'] == expected['input_batches'] == 1
    assert actual['microbatches'] == 4 and expected['microbatches'] == 1
    assert actual['valid_tokens'] == expected['valid_tokens'] == 15
    assert actual['effective_weight'] == pytest.approx(expected['effective_weight'])
    assert actual['loss'] == pytest.approx(expected['loss'], rel=2e-6)
    for a, b in zip(model.parameters(), control.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(a, b, rtol=2e-4, atol=1e-7)


def host_pack(length, *, explicit=False, source_count=1):
    ids = torch.arange(2, length+2).unsqueeze(0)
    result = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  segment_ids=torch.zeros_like(ids, dtype=torch.int32),
                  position_ids=torch.arange(length).unsqueeze(0)+37,
                  labels=ids.clone(), loss_weights=(ids.float()+source_count)/7,
                  _spans=((0, 0, length),), _source_count=source_count)
    result['labels'][0, -2] = -100
    if explicit:
        result.update(prediction_targets=ids.flip(-1), prediction_weights=result['loss_weights']/3,
                      attention_end=torch.full_like(ids, length-1))
    return result


@pytest.mark.parametrize('explicit', [False, True])
def test_accumulated_packs_preserve_targets_positions_and_groups(explicit):
    inputs = [host_pack(n, explicit=explicit, source_count=i+1) for i, n in enumerate((3, 4, 5))]
    before = [{key: value.clone() for key, value in part.items() if torch.is_tensor(value)} for part in inputs]
    expected = [prediction_targets(part) for part in inputs]
    merged, = repack_batches(iter(inputs), 12, 1024**2)
    targets, weights = prediction_targets(merged)
    torch.testing.assert_close(targets, torch.cat([pair[0] for pair in expected], dim=1), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.cat([pair[1] for pair in expected], dim=1), rtol=0, atol=0)
    torch.testing.assert_close(merged['position_ids'], torch.cat([p['position_ids'] for p in inputs], dim=1))
    assert merged['_spans'] == ((0, 0, 3), (0, 3, 7), (0, 7, 12))
    assert merged['_source_count'] == 6
    assert runs(canonical_segments(merged['attention_mask'], merged['segment_ids'])) == merged['_spans']
    assert merged['segment_ids'].dtype == torch.int32
    if explicit:
        for _, start, end in merged['_spans']:
            assert torch.all(merged['attention_end'][0, start:end] == end-1)
    for original, snapshot in zip(inputs, before):
        for key, value in snapshot.items():
            torch.testing.assert_close(original[key], value, rtol=0, atol=0)


def test_accumulation_coalescing_is_bounded_and_keeps_oversized_spans_intact():
    consumed = []
    def stream():
        for length in (3, 3, 7, 2):
            consumed.append(length)
            yield host_pack(length)
    output = repack_batches(stream(), 6, 1024**2)
    first = next(output)
    assert consumed == [3, 3] and first['input_ids'].numel() == 6
    large = next(output)
    assert consumed == [3, 3, 7] and large['_spans'] == ((0, 0, 7),)
    tail, = output
    assert tail['input_ids'].numel() == 2


def test_coalescing_respects_field_boundaries_metadata_and_byte_budget():
    plain, explicit = host_pack(3), host_pack(3, explicit=True)
    assert list(repack_batches([plain, explicit], 6, 1024**2)) == [plain, explicit]
    marked = dict(plain, _opaque=object())
    parts = list(repack_batches([plain, marked], 6, 1024**2))
    assert parts[0] is plain and parts[1] is marked
    for budget, memory in ((0, 1024**2), (6, 1)):
        parts = list(repack_batches([plain, plain], budget, memory))
        assert parts[0] is plain and parts[1] is plain


@pytest.mark.parametrize('explicit', [False, True])
def test_coalesced_weighted_update_matches_original_accumulation(args, tiny_config, explicit):
    args.warmup_steps = 0
    model = CausalLM(tiny_config, Runtime(rank=4, base_dtype=torch.float32, compute_dtype=torch.float32,
                                        attention='vanilla', train_embed=True, train_head=True))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.01)
    control = copy.deepcopy(model)
    data = [host_pack(n, explicit=explicit, source_count=i+1) for i, n in enumerate((3, 4, 5))]
    options = copy.copy(args)
    args.microbatch_token_budget, options.microbatch_token_budget = 12, 0
    actual = Trainer(model, make_optimizer(model, args), args, device='cpu').update(iter(data), step=0, total_steps=2)
    expected = Trainer(control, make_optimizer(control, options), options, device='cpu').update(iter(data), step=0, total_steps=2)
    assert actual['input_batches'] == expected['input_batches'] == 3
    assert actual['microbatches'] == 1 and expected['microbatches'] == 3
    assert actual['valid_tokens'] == expected['valid_tokens'] == 12
    assert actual['effective_weight'] == pytest.approx(expected['effective_weight'])
    assert actual['loss'] == pytest.approx(expected['loss'], rel=2e-6)
    for a, b in zip(model.parameters(), control.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(a, b, rtol=2e-4, atol=1e-7)
