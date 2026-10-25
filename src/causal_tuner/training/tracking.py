"""W&B run identity follows the training checkpoint across invocations."""


def start_wandb(args, tracking, config):
    import wandb
    saved = tracking.get("wandb")
    options = dict(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=config)
    if saved:
        if (args.wandb_project != saved["project"] or
                (args.wandb_entity is not None and args.wandb_entity != saved["entity"])):
            raise ValueError("W&B project and entity must match the run recorded in the checkpoint")
        options.update(id=saved["id"], entity=saved["entity"], resume="must")
    run = wandb.init(**options)
    try:
        # Training may replay updates logged after the most recent checkpoint.
        # Let W&B append events and use the optimizer step only as a chart axis.
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        tracking["wandb"] = dict(id=run.id, project=run.project, entity=run.entity)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run
