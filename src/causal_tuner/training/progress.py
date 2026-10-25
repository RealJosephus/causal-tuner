"""Training progress with compiler time separated from update timing."""

from collections import deque
import sys


class CompilationTimer:
    def __init__(self):
        self.seconds = 0.0
        self.compilation = None

    def start(self):
        import triton
        self.compilation = triton.knobs.compilation
        self.previous = self.compilation.listener
        self.compilation.listener = self.record

    def record(self, **event):
        # Triton reports both fresh compilation and loading compiled disk cache.
        self.seconds += event["times"].total/1e6
        if self.previous is not None:
            self.previous(**event)

    def close(self):
        if self.compilation is not None:
            self.compilation.listener = self.previous
            self.compilation = None


def duration(seconds):
    if seconds is None:
        return "--:--:--"
    hours, seconds = divmod(round(max(0, seconds)), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgress:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.updates = 0
        self.elapsed = 0.0
        self.samples = deque(maxlen=20)

    def snapshot(self, step, *, complete=False):
        mean = sum(self.samples)/len(self.samples) if self.samples else None
        remaining = max(0, self.total_steps-step) if self.total_steps is not None else None
        return {"training_elapsed_seconds": self.elapsed, "mean_step_seconds": mean,
                "eta_seconds": (0.0 if complete else remaining*mean
                                if remaining is not None and mean is not None else None),
                "progress_percent": (100.0 if complete else min(100., 100*step/self.total_steps)
                                     if self.total_steps is not None else None)}

    def update(self, step, elapsed, compilation, *, complete=False):
        training = max(0.0, elapsed-compilation)
        self.elapsed += training
        self.updates += 1
        # Exclude the invocation's first update from ETA to avoid lazy startup
        # costs. Later shape-specific compiler work is subtracted separately.
        if self.updates > 1 and training > 0:
            self.samples.append(training)
        return {"training_seconds": training, "compilation_seconds": compilation,
                **self.snapshot(step, complete=complete)}

    def display(self, step, metrics=None, *, complete=False):
        values = self.snapshot(step, complete=complete)
        count = f"{step}/{self.total_steps}" if self.total_steps is not None else str(step)
        percent = values["progress_percent"]
        fraction = f" ({percent:.1f}%)" if percent is not None else ""
        details = (f" | loss {metrics['loss']:.5g} | {metrics['training_seconds']:.2f}s/step"
                   if metrics is not None else "")
        compile_time = metrics.get("compilation_seconds", 0) if metrics is not None else 0
        if compile_time:
            details += f" | compile {compile_time:.2f}s"
        print(f"Training {count}{fraction}{details} | train {duration(self.elapsed)}"
              f" | ETA {duration(values['eta_seconds'])}", file=sys.stderr, flush=True)
