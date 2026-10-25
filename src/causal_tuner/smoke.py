"""Small-model training, checkpoint, and continuation checks."""

import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import torch
from .arguments import parse_args
from .modeling.config import ModelConfig
from .modeling.model import CausalLM
from .train import runtime_from_args
from .training.engine import Trainer, make_optimizer
from .training.memory import ActivationOffload, GIB
from .checkpointing.io import save_checkpoint, latest_checkpoint, restore_checkpoint, export_adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    options = parser.parse_args(argv)
    if options.offload and options.device != "cuda":
        parser.error("--offload requires --device cuda")
    torch.set_num_threads(2)
    torch.manual_seed(8)
    args = parse_args(["--repo-id", "smoke-fixture", "--dataset-name", "smoke-fixture", "--attn-mechanism", "vanilla",
                       "--dtype", "bfloat16" if options.bf16 else "float32", "--param-dtype", "float32",
                       "--lora-rank", "4", "--lora-train-lm-head", "--lora-train-embed", "--mlp-chunk-size", "3",
                       "--projection-chunk-size", "3", "--remat-group-size", "2", "--loss-token-budget", "2"])
    model = CausalLM(ModelConfig.tiny(), runtime_from_args(args), device=options.device)
    baseline = copy.deepcopy(model)
    if options.offload:
        model.offload = ActivationOffload(72*GIB, 1024**2, 4096)
    optimizer = make_optimizer(model, args)
    trainer = Trainer(model, optimizer, args, device=options.device)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "segment_ids": torch.tensor([[0, 0, 2, 0, 0, 0]]),
             "loss_weights": torch.tensor([[0., .2, 1., 0., .3, .1]])}
    start = time.perf_counter()
    first = trainer.update([batch, batch], step=0, total_steps=2)
    with tempfile.TemporaryDirectory(prefix="causal-tuner-smoke-") as directory:
        save_checkpoint(directory, model, optimizer, step=1, cursor={"epoch": 0, "batch": 2}, signature={"tiny": True},
                        scheduler={"step": 1, "total": 2}, save_optimizer=True)
        second = trainer.update([batch], step=1, total_steps=2)
        resumed_optimizer = make_optimizer(baseline, args)
        restore_checkpoint(*latest_checkpoint(directory, {"tiny": True}), baseline, resumed_optimizer)
        Trainer(baseline, resumed_optimizer, args, device=options.device).update([batch], step=1, total_steps=2)
        for p, q in zip(model.parameters(), baseline.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        export_adapter(model, Path(directory)/"adapter", base_identity="smoke-fixture")
    print(json.dumps({"scope": "small operator shapes only", "device": options.device, "first_update": first,
                      "second_update": second, "exact_resume": True, "elapsed_seconds": time.perf_counter()-start,
                      "checkpoint_cpu_peak_bytes": model.offload.peak if model.offload else 0}, indent=2))


if __name__ == "__main__":
    main()
