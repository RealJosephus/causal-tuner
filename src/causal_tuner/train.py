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
