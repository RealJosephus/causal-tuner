"""Export saved weights independently of training completion or continuation."""

import argparse
import json
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(prog="causal-tune-export", description=__doc__)
    result.add_argument("--checkpoint", required=True, help="Local checkpoint directory or Hub model repository")
    result.add_argument("--checkpoint-revision", help="Hub checkpoint revision; omission resolves the current default branch")
    result.add_argument("--output-dir", required=True, help="New local directory for the standard adapter or model")
    result.add_argument("--format", choices=("adapter", "model"), help="Defaults to the saved run's export format")
    result.add_argument("--device", default="cpu", help="Device for loading weights, e.g. cpu or cuda:0")
    result.add_argument("--repo-id", help="Override the base model location; saved weight identity must still match")
    result.add_argument("--revision", help="Override the saved base model revision")
    result.add_argument("--processor-repo-id", help="Override the tokenizer location; saved identity must still match")
    result.add_argument("--processor-revision", help="Override the saved tokenizer revision")
    result.add_argument("--token", help="Hugging Face token; otherwise use the normal Hub credentials")
    result.add_argument("--host-memory-budget-gib", type=float, help="Override the saved host memory budget")
    result.add_argument("--hf-retry-initial-delay", type=float, default=1)
    result.add_argument("--hf-retry-max-delay", type=float, default=60)
    return result


def run(options):
    import math
    import torch
    from .checkpointing.io import read_manifest, restore_checkpoint, export_adapter, export_hf
    from .checkpointing.huggingface import resolve_assets, base_identity, load_weights
    from .data.tokenizer import TrainingTokenizer
    from .modeling.config import ModelConfig
    from .modeling.model import CausalLM
    from .train import runtime_from_args
    from .training.memory import GIB

    if not (0 < options.hf_retry_initial_delay <= options.hf_retry_max_delay < math.inf):
        raise ValueError("HF retry delays must be finite, positive and ordered")
    source, destination = Path(options.checkpoint), Path(options.output_dir)
    if destination.exists():
        raise FileExistsError("Export destination already exists")
    remote = None
    if source.is_dir():
        if options.checkpoint_revision is not None:
            raise ValueError("Checkpoint revisions apply only to Hub repositories")
        if destination.resolve().is_relative_to(source.resolve()):
            raise ValueError("Export destination must be outside the source checkpoint")
        manifest = read_manifest(source)
    else:
        from .checkpointing.hub import HubCheckpoint
        hub_args = argparse.Namespace(**vars(options), hub_model_repo_id=options.checkpoint)
        remote = HubCheckpoint(hub_args, read_only=True, revision=options.checkpoint_revision)
        checkpoint = remote.load()
        if checkpoint is None:
            raise ValueError("No committed training checkpoint exists at this Hub revision")
        manifest = checkpoint[1]

    saved = manifest["signature"]
    args = argparse.Namespace(**saved["arguments"])
    args.frozen_components = tuple(x.strip() for x in (args.frozen_parameters or "").replace("|", ",").split(",") if x.strip())
    kind = options.format or ("adapter" if args.lora_save_adapter_only else "model")
    if kind == "adapter" and not args.lora:
        raise ValueError("Adapter export requires a LoRA checkpoint")
    budget = args.host_memory_budget_gib if options.host_memory_budget_gib is None else options.host_memory_budget_gib
    if not math.isfinite(budget) or not 0 < budget <= 72:
        raise ValueError("Host memory budget must be in (0, 72] GiB")
    budget *= GIB
    provenance = manifest.get("export", {})
    resolve_options = dict(token=options.token, initial_delay=options.hf_retry_initial_delay,
                           max_delay=options.hf_retry_max_delay)
    base_repo = options.repo_id or args.repo_id
    assets = resolve_assets(base_repo, options.revision or provenance.get("base_revision") or args.revision,
                            **resolve_options)
    if base_identity(assets) != saved["base"]:
        raise ValueError("Export base weights do not match the checkpoint")
    processor_repo = options.processor_repo_id or args.processor_repo_id
    processor_revision = options.processor_revision or provenance.get("processor_revision") or args.processor_revision
    processor = (resolve_assets(processor_repo or base_repo, processor_revision, weights=False, **resolve_options)
                 if processor_repo or options.processor_revision or args.processor_revision else assets)
    if provenance.get("tokenizer") and TrainingTokenizer(processor).signature != provenance["tokenizer"]:
        raise ValueError("Export tokenizer does not match the checkpoint")
    metadata = json.loads((assets/"config.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(metadata)
    runtime = runtime_from_args(args)
    model = CausalLM(config, runtime, device="meta")
    load_weights(model, assets, device=torch.device(options.device), quantization=args.quantization, host_budget=budget)
    # Export needs only weights. It never creates an optimizer, restores RNG,
    # opens training data, starts tracking, or writes a training checkpoint.
    restore_options = dict(require_optimizer=False, restore_rng=False, host_budget=budget)
    if remote:
        remote.restore(model, None, **restore_options)
    else:
        restore_checkpoint(source, manifest, model, None, **restore_options)
    if kind == "adapter":
        export_adapter(model, destination, base_identity=base_repo, host_budget=budget, tokenizer_directory=processor)
    else:
        export_hf(model, destination, metadata, host_budget=budget, tokenizer_directory=processor)
    return destination


def main(argv=None):
    options = parser().parse_args(argv)
    print(run(options))


if __name__ == "__main__":
    main()
