import copy
import json
from array import array

import pytest
import torch

from causal_tuner.data.pipeline import DataPipeline, collate, compact, footprint, normalize, pad, pack_group
from causal_tuner.data.segments import prediction_targets
from causal_tuner.modeling.model import CausalLM, Runtime
from causal_tuner.training.engine import Trainer, make_optimizer


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compaction_preserves_targets_attention_and_update(args, tiny_config, explicit, device):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        pytest.importorskip("triton")
    raw = [dict(input_ids=[2, 3, 4, 5, 6, 7], attention_mask=[1, 1, 0, 1, 1, 0],
                segment_ids=[0]*6, position_ids=[1999998, 1999999, 0, 5, 6, 0],
                labels=[2, 3, -100, 5, 6, -100], loss_weights=[0., .2, 0., 0., .7, 0.]),
           dict(input_ids=[8, 9, 10], loss_weights=[0., .5, 1.], labels=[8, -100, 10])]
    if explicit:
        raw[0].update(prediction_targets=[3, 4, -100, 6, 7, -100], prediction_weights=[.3, .2, 0., .1, .7, 0.],
                      attention_end=[1, 1, 2, 4, 4, 5])
        raw[1].update(prediction_targets=[9, -100, 8], prediction_weights=[.2, 0., .8], attention_end=[0, 2, 2])
    records = [compact({**normalize(record, 9, 97), "_source_count": 1}) for record in raw]
    padded = [pad(record, 9, 1) for record in records]
    baseline = {key: torch.tensor([list(record[key]) for record in padded],
                                  dtype=torch.float32 if key.endswith("weights") else torch.long)
                for key in padded[0] if not key.startswith("_")}
    combined = collate(records, 1024**2)
    assert combined["input_ids"].shape == (1, 7)
    assert combined["_spans"] == ((0, 0, 2), (0, 2, 4), (0, 4, 7))
    assert combined["_source_count"] == 2
    if explicit:
        assert combined["attention_end"].tolist() == [[1, 1, 3, 3, 4, 6, 6]]
    selected = baseline["attention_mask"].bool()
    for original, actual in zip(prediction_targets(baseline), prediction_targets(combined)):
        torch.testing.assert_close(original[selected], actual.flatten(), rtol=0, atol=0)
    model = CausalLM(tiny_config, Runtime(rank=4, base_dtype=torch.float32, compute_dtype=torch.float32,
                                        attention="vanilla", train_embed=True, train_head=True), device=device)
    control = copy.deepcopy(model)
    on_device = lambda data: {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
    torch.testing.assert_close(model(on_device(baseline))[selected.to(device)], control(on_device(combined))[0], rtol=2e-5, atol=2e-6)
    metrics = Trainer(model, make_optimizer(model, args), args, device=device).update([baseline], step=0, total_steps=2)
    compact_metrics = Trainer(control, make_optimizer(control, args), args, device=device).update([combined], step=0, total_steps=2)
    assert metrics["effective_weight"] == pytest.approx(compact_metrics["effective_weight"])
    assert metrics["loss"] == pytest.approx(compact_metrics["loss"])
    for p, q in zip(model.parameters(), control.parameters()):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(p, q, rtol=2e-5, atol=1e-7)


def test_compact_buffers_preserve_values_and_account_for_storage():
    record = normalize({"input_ids": [2, 3, 4]*4096, "loss_weights": [.1, .2, .3]*4096}, 131072, 97)
    packed = compact(record)
    assert all(isinstance(values, array) for values in packed.values())
    assert all(list(packed[key]) == values for key, values in record.items())
    assert footprint(packed) < footprint(record)/4
    assert footprint(packed) >= sum(len(v)*v.itemsize for v in packed.values())
    assert len(pack_group([packed], 131072, 1)[0]["input_ids"]) == 12288
    packed["_source_count"] = 1
    with pytest.raises(MemoryError, match="materialized batch"):
        collate([packed], 1024)


def test_real_long_record_passes_default_shuffle_and_packing_budget(tmp_path, args, tokenizer):
    length = 393216
    source = tmp_path/"long.jsonl"
    source.write_text(json.dumps({"text": "one "*length}))
    args.dataset_name, args.max_sequence_length = str(source), length
    args.packing = args.shuffle = True
    pipeline = DataPipeline(args, tokenizer, 97)
    record = next(pipeline.examples(0))
    assert footprint(record) < pipeline.stage_budget//2
    assert pipeline.stage_budget//footprint(record) >= 7
    batch = next(pipeline.batches())
    assert batch["input_ids"].shape == (1, length)
    assert batch["_spans"] == ((0, 0, length),)
    assert batch["_source_count"] == 1
    assert prediction_targets(batch)[1].sum() == length-1


@pytest.mark.parametrize("packing", [False, True])
def test_short_records_do_not_allocate_global_context(tmp_path, args, tokenizer, packing):
    source = tmp_path/"short.jsonl"
    source.write_text('\n'.join(json.dumps({"text": text}) for text in ["one two", "two three one", "one three"]))
    args.dataset_name, args.max_sequence_length = str(source), 131072
    args.total_batch_size, args.packing = 2, packing
    batches = list(DataPipeline(args, tokenizer, 97).batches())
    assert sum(batch["input_ids"].numel() for batch in batches) == 7
    assert sum(len(batch["_spans"]) for batch in batches) == 3
