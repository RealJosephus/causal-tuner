"""Automatic capacity follows CUDA while explicit memory limits remain enforceable."""

import sys
from types import SimpleNamespace

import pytest
import torch

from causal_tuner.arguments import parse_args
from causal_tuner.training import memory


def test_gpu_and_checkpoint_budget_options():
    flags = ["--repo-id", "random", "--dataset-name", "random"]
    assert parse_args(flags).gpu_memory_budget_gib == 0
    options = parse_args(flags+["--gpu-memory-budget-gib", "79.5", "--checkpoint-host-budget-gib", "64"])
    assert options.gpu_memory_budget_gib == 79.5 and options.checkpoint_host_budget_gib == 64
    options = parse_args(flags+["--host-memory-budget-gib", "256", "--checkpoint-host-budget-gib", "128",
                               "--pinned-memory-budget-gib", "64"])
    assert options.host_memory_budget_gib == 256
    assert options.checkpoint_host_budget_gib == 128 and options.pinned_memory_budget_gib == 64
    for extra in (["--gpu-memory-budget-gib", "-1"], ["--gpu-memory-budget-gib", "nan"],
                  ["--checkpoint-host-budget-gib", "73"]):
        with pytest.raises(SystemExit):
            parse_args(flags+extra)


@pytest.mark.parametrize("platform, name, capacity, visible", [
    ("linux", "NVIDIA GeForce RTX 3060 Laptop GPU", 6, 1),
    ("win32", "NVIDIA GeForce RTX 3060 Laptop GPU", 6, 1),
    ("linux", "NVIDIA H100 80GB HBM3", 80, 2),
    ("win32", "NVIDIA A100 40GB", 40, 2),
])
@pytest.mark.parametrize("explicit", [0, 78])
def test_guard_has_no_platform_hardware_or_estimated_capacity_gate(monkeypatch, platform, name, capacity, visible, explicit):
    gib = memory.GIB
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name=name, total_memory=capacity*gib))
    def unexpected_capacity_query(*args):
        raise AssertionError("Startup must leave capacity to actual allocations and explicit limits")
    monkeypatch.setattr(torch.cuda, "mem_get_info", unexpected_capacity_query)
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(total=8*gib))
    options = SimpleNamespace(gpu_memory_budget_gib=explicit, host_memory_budget_gib=72)
    assert memory.guard_device(options, "cuda:0") == (explicit*gib if explicit else None)


def test_guard_requires_cuda():
    with pytest.raises(RuntimeError, match="CUDA device"):
        memory.guard_device(SimpleNamespace(gpu_memory_budget_gib=0), "cpu")


def test_peak_measurement_preserves_explicit_limit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 75*memory.GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 78*memory.GIB)
    assert memory.check_gpu(None) == {"peak_allocated_gib": 75, "peak_reserved_gib": 78}
    assert memory.check_gpu(79*memory.GIB)["peak_reserved_gib"] == 78
    with pytest.raises(MemoryError, match="configured budget"):
        memory.check_gpu(72*memory.GIB)
