"""Official Hub repository APIs with explicit access settings and atomic commits."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

from .hub import retry


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True)+"\n").encode("utf-8")


class HubRepository:
    def __init__(self, repo_id, repo_type, args, *, read_only=False, revision=None):
        from huggingface_hub import HfApi
        self.repo_id, self.repo_type, self.token = repo_id, repo_type, args.token
        self.api = HfApi(token=args.token)
        self.options = dict(initial_delay=args.hf_retry_initial_delay, max_delay=args.hf_retry_max_delay)
        self.read_only = read_only
        if read_only:
            self.revision = self.call(lambda: self.api.repo_info(repo_id, repo_type=repo_type, revision=revision)).sha
            return
        # No content is uploaded until both settings have been verified.
        self.call(lambda: self.api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True))
        gated = False if args.hub_gated == "off" else args.hub_gated
        self.call(lambda: self.api.update_repo_settings(repo_id, repo_type=repo_type,
                                                       private=args.hub_private, gated=gated))
        info = self.info()
        if info.private != args.hub_private or info.gated != gated:
            raise ValueError("Hub repository access settings were not applied; refusing to upload")
        self.revision = info.sha

    def call(self, operation):
        return retry(operation, **self.options)

    def info(self):
        return self.call(lambda: self.api.repo_info(self.repo_id, repo_type=self.repo_type))

    @contextmanager
    def download(self, filename, revision):
        from huggingface_hub import hf_hub_download
        # local_dir bypasses the persistent snapshot cache. A checkpoint file
        # stays available through restore and is released afterwards.
        with tempfile.TemporaryDirectory(prefix="causal-tuner-hub-") as directory:
            path = self.call(lambda: hf_hub_download(self.repo_id, filename, repo_type=self.repo_type,
                             revision=revision, token=self.token, local_dir=directory))
            yield Path(path)

    def read(self, filename, revision):
        from huggingface_hub.errors import EntryNotFoundError
        try:
            with self.download(filename, revision) as path:
                return path.read_bytes()
        except EntryNotFoundError:
            return None

    def preupload(self, name, content):
        if self.read_only:
            raise ValueError("This Hub repository was opened for reading only")
        from huggingface_hub import CommitOperationAdd
        operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=content)
        self.call(lambda: self.api.preupload_lfs_files(self.repo_id, repo_type=self.repo_type,
                  additions=[operation], free_memory=True))
        return operation

    def commit(self, operations, *, parent, marker, content, message):
        if self.read_only:
            raise ValueError("This Hub repository was opened for reading only")
        from huggingface_hub import CommitOperationAdd
        operations = [*operations, CommitOperationAdd(path_in_repo=marker, path_or_fileobj=content)]
        committed = self.call(lambda: self.api.create_commit(
            self.repo_id, repo_type=self.repo_type, operations=operations,
            parent_commit=parent, commit_message=message))
        self.revision = committed.oid
        return self.revision

    def url(self, revision=None):
        prefix = "datasets/" if self.repo_type == "dataset" else ""
        return f"https://huggingface.co/{prefix}{self.repo_id}/commit/{revision or self.revision}"
