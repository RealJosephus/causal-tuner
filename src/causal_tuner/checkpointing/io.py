"""Filesystem integrity helpers for dataset manifests."""

import hashlib
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, name):
    path = (root/name).resolve()
    if path.parent != root.resolve():
        raise ValueError("Manifest contains a path outside its checkpoint directory")
    return path
