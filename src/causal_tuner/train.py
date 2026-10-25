"""Single-device training entry point. Help does not import or initialize CUDA."""

from .arguments import parse_args


def training_plan(args, pipeline):
    import math
    if args.max_steps > 0:
        return {"steps": args.max_steps, "batches": None, "source_per_epoch": None}
    if args.dataset_streaming:
        return {"steps": None, "batches": None, "source_per_epoch": pipeline.source_count}
    # Preparation has already filtered and packed the stored records.
    prepared_batches = ((pipeline.source_count+args.total_batch_size-1)//args.total_batch_size
                        if args.dataset_format == "prepared" else None)
    batches = [math.ceil((prepared_batches if prepared_batches is not None else
                          sum(1 for _ in pipeline.batches(epoch)))*min(1., args.num_train_epochs-epoch))
               for epoch in range(math.ceil(args.num_train_epochs))]
    return {"steps": sum(math.ceil(n/args.gradient_accumulation_steps) for n in batches),
            "batches": batches, "source_per_epoch": None}


def streaming_position(plan, cursor):
    count = plan["source_per_epoch"]
    return cursor["epoch"]*count+min(cursor["source_examples"], count)


def streaming_schedule(args, plan, cursor, step):
    import math
    count = plan["source_per_epoch"]
    if count is None:
        return step, None, args.warmup_steps
    position = streaming_position(plan, cursor)
    horizon = math.ceil(count*args.num_train_epochs*1.05)
    start = cursor.get("warmup_source_position", 0)
    # Warmup is measured in completed updates. Decay starts at its committed
    # source position and retains the fixed, metadata-based epoch horizon.
    return step, None, args.warmup_steps, (position-start)/max(1, horizon-start)


def runtime_from_args(args):
    import torch
    from .modeling.model import Runtime
    return Runtime(rank=args.lora_rank if args.lora else 0,
                   base_dtype=getattr(torch, args.param_dtype), compute_dtype=getattr(torch, args.dtype),
                   train_dtype=torch.float32 if args.precision_recipe == "stable" else torch.bfloat16,
                   train_embed=args.lora_train_embed, train_head=args.lora_train_lm_head, frozen=args.frozen_components,
                   attention=args.attn_mechanism,
                   mlp_tile=args.mlp_chunk_size, projection_tile=args.projection_chunk_size,
                   mlp_cache_budget_bytes=None if args.mlp_cache_budget_gib is None else int(args.mlp_cache_budget_gib*1024**3),
                   attention_cache_budget_bytes=None if args.attention_cache_budget_gib is None else int(args.attention_cache_budget_gib*1024**3),
                   adaptive=args.gradient_checkpointing == "auto",
                   checkpoint=args.gradient_checkpointing != "disabled", group_size=args.remat_group_size,
                   inner_checkpoint=args.remat_inner_checkpoint)


def run(args):
    import itertools
    import json
    import random
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from .modeling.config import ModelConfig
    from .modeling.model import CausalLM
    from .data.tokenizer import TrainingTokenizer
    from .data.pipeline import DataPipeline
    from .data.maskgit import MaskGIT
    from .checkpointing.huggingface import resolve_assets, load_weights, base_identity
    from .checkpointing.io import latest_checkpoint, restore_checkpoint, save_checkpoint, export_hf, export_adapter, GIB
    from .training.engine import Trainer, make_optimizer, learning_rate
    from .training.memory import ActivationOffload, guard_device, check_host, check_gpu, host_peak_bytes
    from .training.progress import CompilationTimer, TrainingProgress
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production training; CPU operator tests are in tests/")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    assets = resolve_assets(args.repo_id, args.revision, args.token, weights=False,
                             initial_delay=args.hf_retry_initial_delay, max_delay=args.hf_retry_max_delay)
    metadata = json.loads((assets/"config.json").read_text())
    config = ModelConfig.from_dict(metadata)
    budget = guard_device(args, device)
    from .kernels.probe import probe_backends
    probe_backends(args, device)
    assets = resolve_assets(args.repo_id, args.revision if Path(args.repo_id).is_dir() else assets.name, args.token, weights=True,
                            initial_delay=args.hf_retry_initial_delay, max_delay=args.hf_retry_max_delay)
    processor = (resolve_assets(args.processor_repo_id or args.repo_id, args.processor_revision,
                                args.token, weights=False, initial_delay=args.hf_retry_initial_delay,
                                max_delay=args.hf_retry_max_delay)
                 if args.processor_repo_id or args.processor_revision else assets)
    tokenizer = TrainingTokenizer(processor)
    export_source = {"base_revision": None if Path(args.repo_id).is_dir() else assets.name,
                     "processor_revision": None if Path(args.processor_repo_id or args.repo_id).is_dir() else processor.name,
                     "tokenizer": tokenizer.signature}
    if max(tokenizer.get_vocab().values()) >= config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds the model embedding table")
    pipeline = DataPipeline(args, tokenizer, config.vocab_size)
    signature_args = {k: v for k, v in vars(args).items() if k not in {
        "token", "output_dir", "logging_steps", "save_steps", "save_total_limit", "save_optimizer_state",
        "use_wandb", "wandb_entity", "wandb_project", "wandb_run_name", "track_memory", "frozen_components", "stop_after_updates",
        "hub_dataset_repo_id", "hub_model_repo_id", "hub_private", "hub_gated"}}
    import importlib.metadata
    import platform
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for dependency in ("bitsandbytes", "triton", "triton-windows", "flash-attn", "cut-cross-entropy", "safetensors",
                       "tokenizers", "datasets", "numpy", "jinja2", "huggingface-hub", "ijson", "pyarrow"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    signature = json.loads(json.dumps({"base": base_identity(assets), "data": pipeline.identity,
                                       "arguments": signature_args, "runtime": versions, "format": 1}))
    remote = None
    if args.hub_model_repo_id:
        from .checkpointing.hub import HubCheckpoint
        remote = HubCheckpoint(args)
    checkpoint = remote.latest(signature) if remote else latest_checkpoint(args.output_dir, signature)
    plan = checkpoint[1]["scheduler"]["plan"] if checkpoint else training_plan(args, pipeline)
    total_steps, plans = plan["steps"], plan["batches"]
    schedule = ((0, total_steps) if total_steps is not None else
                streaming_schedule(args, plan, dict(epoch=0, source_examples=0), 0))
    learning_rate(args, *schedule)
    complete = bool(checkpoint and checkpoint[1]["scheduler"]["complete"])
    if complete:
        kind = "adapter" if args.lora_save_adapter_only else "export"
        exported = (remote.export_complete(checkpoint[1]["step"], signature, kind) if remote else
                    (Path(args.output_dir)/f"{kind}-{checkpoint[1]['step']:08d}").is_dir())
        if exported:
            print("The requested training and export are already complete.")
            return
    if checkpoint and not complete and not checkpoint[1]["optimizer_saved"]:
        raise ValueError("The latest compatible checkpoint has no optimizer state; exact continuation is unavailable")
    runtime = runtime_from_args(args)
    runtime.gpu_budget_bytes = None if budget is None else int(budget)
    model = CausalLM(config, runtime, device="meta")
    load_weights(model, assets, device=device, quantization=args.quantization, host_budget=args.host_memory_budget_gib*GIB)
    if args.activation_offload == "cpu" or (args.activation_offload == "auto" and runtime.checkpoint):
        model.offload = ActivationOffload(args.host_memory_budget_gib*GIB, args.checkpoint_host_budget_gib*GIB,
                                          args.pinned_memory_budget_gib*GIB, gpu_budget=budget)
    optimizer = make_optimizer(model, args)
    trainer = Trainer(model, optimizer, args, device=device)
    step, cursor = 0, {"epoch": 0, "batch": 0, "micro_step": 0, "source_examples": 0}
    if checkpoint:
        if remote:
            step, cursor = remote.restore(model, None if complete else optimizer, require_optimizer=not complete)
        else:
            step, cursor = restore_checkpoint(*checkpoint, model, None if complete else optimizer, require_optimizer=not complete)
    print(json.dumps({"phase": "loaded", **check_gpu(budget), "host_peak_gib": host_peak_bytes()/GIB}))
    maskgit = MaskGIT(tokenizer, args.maskgit_probability, args.seed) if args.maskgit_probability else None
    logger = None
    tracking = dict(checkpoint[1].get("tracking", {})) if checkpoint else {}
    print(json.dumps({"device": torch.cuda.get_device_name(device), "trainable_dtype": str(model.runtime.train_dtype),
                      "compute_dtype": args.dtype, "attention": args.attn_mechanism,
                      "loss": args.loss_implementation,
                      "gpu_budget_gib": None if budget is None else budget/GIB, "total_steps": total_steps,
                      "source_examples_per_epoch_estimate": plan["source_per_epoch"]}))
    last_saved = checkpoint[1]["step"] if checkpoint else -1
    invocation_updates = 0
    def save():
        nonlocal last_saved
        if last_saved == step:
            return
        options = dict(step=step, cursor=cursor, signature=signature,
                       scheduler={"step": step, "plan": plan, "complete": complete},
                       save_optimizer=args.save_optimizer_state, host_budget=args.host_memory_budget_gib*GIB,
                       tracking=tracking, export=export_source)
        location = (remote.save(model, optimizer, **options) if remote else
                    str(save_checkpoint(args.output_dir, model, optimizer, keep=args.save_total_limit, **options)))
        last_saved = step
        print(json.dumps({"phase": "checkpoint_saved", "step": step, "location": location, **check_gpu(budget),
                          "host_peak_gib": host_peak_bytes()/GIB}))
    failed = True
    compiler = CompilationTimer()
    progress = TrainingProgress(total_steps)
    try:
        compiler.start()
        if args.use_wandb:
            from .training.tracking import start_wandb
            logger = start_wandb(args, tracking, {**signature_args, "runtime": versions})
        progress.display(step, complete=complete)
        while not complete:
            epoch, skipped = cursor["epoch"], cursor["batch"]
            batches = pipeline.batches(epoch, skip=skipped)
            if plans is not None:
                batches = itertools.islice(batches, max(0, plans[epoch]-skipped))
            for_update = iter(batches)
            first = next(for_update, None)
            if first is None:
                raise ValueError("Checkpoint batch cursor is beyond its epoch")
            while first is not None:
                start = time.perf_counter()
                compiled_before = compiler.seconds
                group_count, source_count = 0, 0
                def update_batches():
                    nonlocal group_count, source_count, first
                    source = itertools.chain((first,), itertools.islice(for_update, args.gradient_accumulation_steps-1))
                    first = None
                    for b in source:
                        value = maskgit(b, cursor["micro_step"]+group_count) if maskgit else b
                        group_count += 1
                        source_count += b["_source_count"]
                        yield value
                if args.track_memory:
                    torch.cuda.reset_peak_memory_stats()
                schedule = streaming_schedule(args, plan, cursor, step) if total_steps is None else None
                metrics = trainer.update(update_batches(), step=step, total_steps=total_steps, schedule=schedule)
                cursor["batch"] += group_count
                cursor["micro_step"] += group_count
                cursor["source_examples"] += source_count
                step += 1
                if total_steps is None and plan["source_per_epoch"] is not None and step == args.warmup_steps:
                    cursor["warmup_source_position"] = streaming_position(plan, cursor)
                invocation_updates += 1
                torch.cuda.synchronize()
                metrics.update(step=step, elapsed_seconds=time.perf_counter()-start,
                               host_rss_gib=check_host(args.host_memory_budget_gib*GIB)/GIB,
                               epoch=epoch, source_examples=cursor["source_examples"])
                metrics["tokens_per_second"] = metrics["valid_tokens"]/metrics["elapsed_seconds"]
                gpu_peaks = check_gpu(budget)
                if args.track_memory:
                    metrics.update(**gpu_peaks, host_peak_gib=host_peak_bytes()/GIB,
                                   checkpoint_host_gib=(model.offload.peak/GIB if model.offload else 0),
                                   pinned_peak_gib=(model.offload.pinned_peak/GIB if model.offload else 0),
                                   offload=(dict(model.offload.stats) if model.offload else None))
                complete = total_steps is not None and step >= total_steps
                # One bounded lookahead resolves real EOF before committing the cursor.
                first = None if complete else next(for_update, None)
                if first is None and not complete:
                    cursor.update(epoch=epoch+1, batch=0, source_examples=0)
                    complete = total_steps is None and cursor["epoch"] >= args.num_train_epochs
                metrics.update(progress.update(step, metrics["elapsed_seconds"], compiler.seconds-compiled_before,
                                               complete=complete))
                progress.display(step, metrics, complete=complete)
                if step % args.logging_steps == 0 or complete:
                    print(json.dumps(metrics), flush=True)
                    if logger:
                        logger.log(metrics)
                if args.save_steps and step % args.save_steps == 0:
                    save()
                if complete or (args.stop_after_updates and invocation_updates >= args.stop_after_updates):
                    break
            if args.stop_after_updates and invocation_updates >= args.stop_after_updates:
                break
        save()
        optimizer.zero_grad(set_to_none=True)
        if not complete:
            print("Stopped after the requested invocation updates. Resume with the same training schedule.")
        else:
            from contextlib import nullcontext
            import tempfile
            kind = "adapter" if args.lora_save_adapter_only else "export"
            # Final model export uses scratch space in Hub mode; periodic
            # resumable checkpoints themselves are streamed from memory.
            with (tempfile.TemporaryDirectory(prefix="causal-tuner-export-") if remote else
                  nullcontext(args.output_dir)) as directory:
                destination = Path(directory)/f"{kind}-{step:08d}"
                if not destination.exists():
                    if args.lora_save_adapter_only:
                        export_adapter(model, destination, base_identity=args.repo_id,
                                       host_budget=args.host_memory_budget_gib*GIB, tokenizer_directory=processor)
                    else:
                        export_hf(model, destination, metadata, host_budget=args.host_memory_budget_gib*GIB,
                                  tokenizer_directory=processor)
                if remote:
                    print(json.dumps({"phase": "export_uploaded", "location": remote.publish_export(
                        destination, step=step, signature=signature, kind=kind)}))
        failed = False
    finally:
        compiler.close()
        if model.offload is not None:
            model.offload.close()
        if logger:
            logger.finish(exit_code=1 if failed else 0)


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
