"""Real FA2 retention and replay; a layout substitute is insufficient here."""

import copy
import functools
import pytest
import torch
from causal_tuner.modeling.model import CausalLM, Runtime
from conftest import batch


@pytest.mark.parametrize('packed', [False, True])
def test_attention_retention_matches_gradients_and_repeated_backward(tiny_config, monkeypatch, packed):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    fa = pytest.importorskip('flash_attn')
    from flash_attn import flash_attn_interface as interface
    for name in ('flash_attn_func', 'flash_attn_varlen_func'):
        monkeypatch.setattr(fa, name, functools.partial(getattr(fa, name), deterministic=True))
    calls = {'forward': 0}
    name = 'varlen_fwd' if packed else 'fwd'
    original = getattr(interface.flash_attn_gpu, name)
    def counted(*args, **kwargs):
        calls['forward'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(interface.flash_attn_gpu, name, counted)
    runtime = Runtime(rank=4, mlp_tile=5, checkpoint=True, attention_cache_budget_bytes=1024**2)
    model = CausalLM(tiny_config, runtime, 'cuda')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.01)
    reference = copy.deepcopy(model)
    reference.runtime.attention_cache_budget_bytes = 0
    data = batch([[2, 3, 4, 5, 6, 7, 8]], segments=[[0, 0, 0, 1, 1, 1, 1]] if packed else None)
    data = {key: value.cuda() for key, value in data.items()}
    actual = model(data)
    signal = torch.randn_like(actual)
    parameters = [p for p in model.parameters() if p.requires_grad]
    first = torch.autograd.grad(actual, parameters, signal, retain_graph=True)
    assert calls['forward'] == 2
    second = torch.autograd.grad(actual, parameters, signal)
    assert calls['forward'] == 4
    expected = reference(data)
    gradients = torch.autograd.grad(expected, [p for p in reference.parameters() if p.requires_grad], signal)
    assert calls['forward'] == 8
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b, c in zip(first, second, gradients):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)


def test_checkpoint_captures_tile_before_interleaved_forward(tiny_config):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    pytest.importorskip('triton')
    runtime = Runtime(rank=4, attention='vanilla', mlp_tile=5)
    model = CausalLM(tiny_config, runtime, 'cuda')
    reference = copy.deepcopy(model)
    def data(length):
        return {key: value.cuda() for key, value in batch([[2]*length]).items()}
    first = model(data(11))
    runtime.mlp_tile, runtime.checkpoint = 7, False
    second = model(data(13))
    first.sum().backward()
    reference(data(11)).sum().backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    second.sum().backward()


@pytest.mark.parametrize('accumulated', [False, True])
def test_real_fa2_repacked_weighted_gradient(args, monkeypatch, accumulated):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    fa = pytest.importorskip('flash_attn')
    from dataclasses import replace
    from causal_tuner.data.pipeline import collate, compact, normalize
    from causal_tuner.modeling.config import ModelConfig
    from causal_tuner.training.engine import Trainer, make_optimizer
    for name in ('flash_attn_func', 'flash_attn_varlen_func'):
        monkeypatch.setattr(fa, name, functools.partial(getattr(fa, name), deterministic=True))
    config = replace(ModelConfig(), hidden_size=512, intermediate_size=1536, vocab_size=4096,
                     num_hidden_layers=4, num_attention_heads=4, pad_token_id=0, eos_token_id=1)
    model = CausalLM(config, Runtime(rank=32, checkpoint=True, attention_cache_budget_bytes=64*1024**2), 'cuda')
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_(std=.02)
        model.lm_head.weight.normal_(std=.02)
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.005)
    reference = copy.deepcopy(model)
    records = []
    for i, length in enumerate((733, 1901, 3101, 4093)):
        record = normalize(dict(input_ids=torch.randint(2, 4096, (length,)).tolist(),
                                loss_weights=[(.5, 1., 2., 1.)[i]]*length), 8192, 4096)
        records.append(compact(dict(record, _source_count=1)))
    data = collate(records, 64*1024**2)
    options = copy.copy(args)
    args.microbatch_token_budget, options.microbatch_token_budget = (4096 if accumulated else 2048), 0
    args.loss_token_budget = options.loss_token_budget = 128
    batches = [collate([record], 64*1024**2) for record in records] if accumulated else [data]
    actual = Trainer(model, make_optimizer(model, args), args, device='cuda').update(batches, step=0, total_steps=2)
    expected = Trainer(reference, make_optimizer(reference, options), options, device='cuda').update([data], step=0, total_steps=2)
    assert actual['valid_tokens'] == expected['valid_tokens']
    assert actual['effective_weight'] == expected['effective_weight']
    if accumulated:
        assert actual['input_batches'] == 4 and actual['microbatch_rows'] == [2634, 3101, 4093]
    assert abs(actual['loss']-expected['loss']) < .01
    errors = [(a.grad-b.grad).norm()/b.grad.norm().clamp_min(1e-20)
              for a, b in zip(model.parameters(), reference.parameters()) if a.requires_grad]
    assert max(float(value) for value in errors) < .025
