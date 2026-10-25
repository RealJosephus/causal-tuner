"""Checkpoint-owned W&B identity and logging across training restarts."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from causal_tuner.training.tracking import start_wandb
from test_hub_storage import hub
from test_runner import runner_assets


@pytest.fixture
def wandb_service(monkeypatch):
    service = SimpleNamespace(calls=[], runs={}, fail_define=False)

    def initialize(**options):
        service.calls.append(options)
        identity = options.get("id", f"run-{len(service.runs)}")
        if options.get("resume") == "must":
            assert identity in service.runs
        state = service.runs.setdefault(identity, SimpleNamespace(logs=[], definitions=[], exits=[]))

        def define_metric(name, **kwargs):
            if service.fail_define:
                raise RuntimeError("metric definition failed")
            state.definitions.append((name, kwargs))

        def log(metrics, **kwargs):
            # W&B's append-only event step must not be set to a replayed
            # optimizer step, which can be earlier than its server history.
            assert kwargs == {}
            state.logs.append(copy.deepcopy(metrics))

        return SimpleNamespace(id=identity, project=options["project"], entity=options["entity"] or "account",
                               define_metric=define_metric, log=log,
                               finish=lambda *, exit_code: state.exits.append(exit_code))

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=initialize))
    return service


def test_wandb_identity_resumes_without_local_state(args, wandb_service):
    args.wandb_entity, args.wandb_run_name = "team", "experiment"
    tracking = {}
    first = start_wandb(args, tracking, {"learning_rate": args.learning_rate})
    assert tracking == {"wandb": {"id": first.id, "project": "causal-tuner", "entity": "team"}}
    args.wandb_entity = None  # Resume must keep the checkpoint's team.
    resumed = start_wandb(args, json.loads(json.dumps(tracking)), {})
    assert resumed.id == first.id
    assert wandb_service.calls[1]["resume"] == "must"
    assert wandb_service.calls[1]["entity"] == "team"
    assert ("*", {"step_metric": "step"}) in wandb_service.runs[first.id].definitions
    args.wandb_project = "different"
    with pytest.raises(ValueError, match="project and entity"):
        start_wandb(args, tracking, {})
    args.wandb_project, args.wandb_entity = "causal-tuner", "different-team"
    with pytest.raises(ValueError, match="project and entity"):
        start_wandb(args, tracking, {})
    assert len(wandb_service.calls) == 2


def test_wandb_setup_failure_closes_failed_run(args, wandb_service):
    wandb_service.fail_define = True
    with pytest.raises(RuntimeError, match="metric definition"):
        start_wandb(args, {}, {})
    assert wandb_service.runs["run-0"].exits == [1]


def test_real_wandb_offline_accepts_replayed_training_steps(args, monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    monkeypatch.setenv("WANDB_DISABLE_CODE", "true")
    tracking = {}
    logger = start_wandb(args, tracking, {"learning_rate": args.learning_rate})
    try:
        logger.log({"step": 3, "loss": 1.0})
        logger.log({"step": 2, "loss": 2.0})
        assert logger.step == 2
        assert logger.summary["step"] == 2 and logger.summary["loss"] == 2.0
        assert tracking["wandb"]["id"] == logger.id
    finally:
        logger.finish(exit_code=0)


@pytest.mark.parametrize("remote", [False, True])
def test_runner_resumes_wandb_and_replays_steps_after_unsaved_logs(
        args, runner_assets, hub, wandb_service, monkeypatch, tmp_path, remote):
    from causal_tuner.train import run
    from causal_tuner.checkpointing import io
    from causal_tuner.checkpointing.hub import HubCheckpoint

    args.output_dir = str(tmp_path/"output")
    args.hub_model_repo_id = "owner/tracked-run" if remote else None
    args.use_wandb, args.wandb_entity = True, "team"
    args.max_steps, args.stop_after_updates = 3, 1
    args.save_optimizer_state, args.save_steps = True, 1
    args.logging_steps, args.track_memory = 1, True
    args.token = "fixture-secret-not-for-wandb"

    def manifest(step):
        if remote:
            repo = hub.repos["model", args.hub_model_repo_id]
            return json.loads(repo["versions"][repo["sha"]]["checkpoint/manifest.json"])
        return json.loads((Path(args.output_dir)/f"checkpoint-{step:08d}"/"manifest.json").read_text())

    run(args)
    first = manifest(1)
    identity = first["tracking"]["wandb"]
    assert identity == {"id": "run-0", "project": "causal-tuner", "entity": "team"}
    target, name = (HubCheckpoint, "save") if remote else (io, "save_checkpoint")
    save = getattr(target, name)

    def interrupted(*unused, **kwargs):
        raise RuntimeError("interrupted checkpoint upload")

    monkeypatch.setattr(target, name, interrupted)
    with pytest.raises(RuntimeError, match="interrupted checkpoint"):
        run(args)
    assert manifest(1)["step"] == 1
    monkeypatch.setattr(target, name, save)
    run(args)
    assert manifest(2)["tracking"]["wandb"] == identity
    run(args)
    final = manifest(3)
    assert final["scheduler"]["complete"]
    assert final["tracking"]["wandb"] == identity
    run(args)  # Completed training/export must not create another W&B run.

    state = wandb_service.runs["run-0"]
    assert state.exits == [0, 1, 0, 0]
    assert [row["step"] for row in state.logs] == [1, 2, 2, 3]
    assert len(wandb_service.calls) == 4 and len(wandb_service.runs) == 1
    assert all(call["id"] == "run-0" and call["resume"] == "must" for call in wandb_service.calls[1:])
    assert all("token" not in call["config"] for call in wandb_service.calls)
    assert all(set(("loss", "learning_rate", "grad_norm", "tokens_per_second", "valid_tokens",
                    "elapsed_seconds", "host_rss_gib", "host_peak_gib", "eta_seconds", "progress_percent",
                    "training_seconds", "training_elapsed_seconds", "compilation_seconds")) <= row.keys()
               for row in state.logs)
    assert not remote or not Path(args.output_dir).exists()


def test_runner_keeps_wandb_identity_while_logging_disabled(args, runner_assets, wandb_service, tmp_path):
    from causal_tuner.train import run
    args.output_dir = str(tmp_path/"output")
    args.use_wandb, args.save_optimizer_state = True, True
    args.max_steps, args.stop_after_updates, args.logging_steps = 3, 1, 1
    run(args)
    args.use_wandb = False
    run(args)
    manifest = json.loads((Path(args.output_dir)/"checkpoint-00000002"/"manifest.json").read_text())
    assert manifest["tracking"]["wandb"]["id"] == "run-0"
    args.use_wandb = True
    run(args)
    assert len(wandb_service.runs) == 1
    assert [row["step"] for row in wandb_service.runs["run-0"].logs] == [1, 3]
