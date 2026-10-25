"""Bounded checkpoint transfers with one complete checkpoint per Hub Git commit."""

from contextlib import contextmanager
import json
from pathlib import Path

from ..publishing import HubRepository, json_bytes
from .io import TensorWriter, checkpoint_manifest, file_hash, restore_checkpoint, signatures_match, validate_manifest, GIB


class HubCheckpoint:
    def __init__(self, args, *, read_only=False, revision=None):
        self.repo = HubRepository(args.hub_model_repo_id, "model", args, read_only=read_only, revision=revision)
        self.manifest = None

    def load(self):
        content = self.repo.read("checkpoint/manifest.json", self.repo.revision)
        if content is None:
            return None
        manifest = validate_manifest(json.loads(content))
        self.manifest = manifest
        return None, manifest

    def latest(self, signature):
        checkpoint = self.load()
        if checkpoint and not signatures_match(checkpoint[1]["signature"], signature):
            raise ValueError("No compatible checkpoint exists on the Hub; configuration, runtime, model, or data changed")
        return checkpoint

    def save(self, model, optimizer, *, step, cursor, signature, scheduler, save_optimizer=True,
             host_budget=72*GIB, chunk_bytes=64*1024**2, tracking=None, export=None):
        from huggingface_hub import CommitOperationDelete
        if not save_optimizer:
            raise ValueError("Hub training checkpoints require complete optimizer state")
        operations = []

        def put(name, content):
            operation = self.repo.preupload("checkpoint/"+name, content)
            operations.append(operation)
            return operation.upload_info.sha256.hex()

        with TensorWriter(put=put, host_budget=host_budget, chunk_bytes=chunk_bytes) as writer:
            manifest = checkpoint_manifest(writer, model, optimizer, step=step, cursor=cursor, signature=signature,
                                           scheduler=scheduler, save_optimizer=True, tracking=tracking, export=export)
            # Earlier Git revisions, including their optimizer state, remain intact.
            previous = self.manifest["files"] if self.manifest else {}
            operations.extend(CommitOperationDelete("checkpoint/"+name) for name in previous if name not in writer.files)
            self.repo.commit(operations, parent=self.repo.revision, marker="checkpoint/manifest.json",
                             content=json_bytes(manifest), message=f"Training checkpoint at step {step}")
        self.manifest = manifest
        return self.repo.url()

    def restore(self, model, optimizer, *, require_optimizer=True, restore_rng=True, host_budget=72*GIB):
        manifest, revision = self.manifest, self.repo.revision
        if manifest is None:
            raise ValueError("Load a Hub checkpoint before restoring it")

        @contextmanager
        def resolve(name):
            with self.repo.download("checkpoint/"+name, revision) as path:
                if file_hash(path) != manifest["files"][name]:
                    raise ValueError("Checkpoint checksum mismatch")
                yield path

        return restore_checkpoint(None, manifest, model, optimizer, require_optimizer=require_optimizer, resolve=resolve,
                                  restore_rng=restore_rng, host_budget=host_budget)

    def export_complete(self, step, signature, kind):
        content = self.repo.read("training_export.json", self.repo.revision)
        if content is None:
            return False
        marker = json.loads(content)
        return marker.get("step") == step and signatures_match(marker.get("signature"), signature) and marker.get("kind") == kind

    def publish_export(self, directory, *, step, signature, kind):
        from huggingface_hub import CommitOperationDelete
        directory = Path(directory)
        files, operations = {}, []
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                name = path.relative_to(directory).as_posix()
                files[name] = file_hash(path)
                operations.append(self.repo.preupload(name, str(path)))
        previous = self.repo.read("training_export.json", self.repo.revision)
        for name in json.loads(previous).get("files", {}) if previous else ():
            if name.startswith("checkpoint/") or name.startswith(".") or ".." in Path(name).parts:
                raise ValueError("Invalid previous export path")
            if name not in files:
                operations.append(CommitOperationDelete(name))
        marker = dict(step=step, signature=signature, kind=kind, files=files)
        self.repo.commit(operations, parent=self.repo.revision, marker="training_export.json",
                         content=json_bytes(marker), message=f"Export {kind} at step {step}")
        return self.repo.url()
