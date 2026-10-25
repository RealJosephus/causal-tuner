"""Progress timing excludes compiler work and restarts on continuation."""

import sys
from types import SimpleNamespace

import pytest

from causal_tuner.training.progress import CompilationTimer, TrainingProgress, duration


def test_eta_excludes_compilation_warmup_and_time_outside_updates(capsys):
    progress = TrainingProgress(10)
    progress.display(0)
    warmup = progress.update(1, 125., 100.)
    assert warmup["eta_seconds"] is None
    assert warmup["training_seconds"] == 25.
    second = progress.update(2, 70., 60.)
    assert second["training_elapsed_seconds"] == 35.
    assert second["eta_seconds"] == 80.
    assert second["progress_percent"] == 20.
    third = progress.update(3, 20., 0.)
    assert third["mean_step_seconds"] == 15.
    assert third["eta_seconds"] == 105.
    progress.display(3, {"loss": .25, **third})
    lines = capsys.readouterr().err.splitlines()
    assert "Training 0/10 (0.0%)" in lines[0] and "ETA --:--:--" in lines[0]
    assert "Training 3/10 (30.0%)" in lines[1]
    assert "loss 0.25" in lines[1] and "train 00:00:55 | ETA 00:01:45" in lines[1]


def test_eta_window_resume_unknown_total_and_completion():
    progress = TrainingProgress(100)
    progress.update(1, 1000., 0.)
    progress.update(2, 500., 0.)
    for step in range(3, 23):
        metrics = progress.update(step, 12., 2.)
    assert metrics["mean_step_seconds"] == 10.
    assert metrics["eta_seconds"] == 780.
    resumed = TrainingProgress(100)
    assert resumed.snapshot(22)["progress_percent"] == 22.
    assert resumed.update(23, 30., 10.)["eta_seconds"] is None
    assert resumed.update(24, 15., 5.)["eta_seconds"] == 760.
    unknown = TrainingProgress(None)
    assert unknown.update(1, 4., 0.)["progress_percent"] is None
    assert unknown.update(2, 4., 0.)["eta_seconds"] is None
    finished = unknown.update(3, 4., 0., complete=True)
    assert finished["progress_percent"] == 100. and finished["eta_seconds"] == 0.
    assert progress.update(100, 5., 10., complete=True)["training_seconds"] == 0.
    assert duration(90061) == "25:01:01"


def test_compilation_timer_chains_listener_and_restores_after_failure(monkeypatch):
    events = []
    previous = lambda **event: events.append(event)
    compilation = SimpleNamespace(listener=previous)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(knobs=SimpleNamespace(compilation=compilation)))
    timer = CompilationTimer()
    try:
        timer.start()
        for cache_hit, microseconds in ((False, 2_000_000), (True, 500_000)):
            compilation.listener(times=SimpleNamespace(total=microseconds), cache_hit=cache_hit)
        assert timer.seconds == 2.5
        assert [event["cache_hit"] for event in events] == [False, True]
        with pytest.raises(RuntimeError):
            raise RuntimeError("interrupted update")
    finally:
        timer.close()
    assert compilation.listener is previous
    timer.close()


def test_real_compiler_timer_keeps_the_cached_launch_path(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from causal_tuner.kernels.launch import launch

    @triton.jit
    def increment(X, Y, BLOCK: tl.constexpr):
        indices = tl.arange(0, BLOCK)
        tl.store(Y+indices, tl.load(X+indices)+1)

    x = torch.arange(32, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    previous = triton.knobs.compilation.listener
    timer = CompilationTimer()
    with triton.knobs.cache.scope():
        triton.knobs.cache.dir = str(tmp_path/"compiler-cache")
        try:
            timer.start()
            launch(increment, (1,), x, y, BLOCK=32)
            compiled = timer.seconds
            assert compiled > 0
            launch(increment, (1,), x, y, BLOCK=32)
            assert timer.seconds == compiled
            torch.testing.assert_close(y, x+1)
        finally:
            timer.close()
    assert triton.knobs.compilation.listener is previous
