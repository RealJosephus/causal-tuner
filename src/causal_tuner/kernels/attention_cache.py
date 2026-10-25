"""Selective FA2 retention with normal recomputation on repeated backward."""

from functools import lru_cache
import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


@lru_cache(maxsize=1)
def attention_ops():
    # Import registers the native operations before resolving their overloads.
    from flash_attn import flash_attn_interface  # noqa: F401
    return (torch.ops.flash_attn._flash_attn_forward.default,
            torch.ops.flash_attn._flash_attn_varlen_forward.default)


def attention_cache_contexts():
    operations = attention_ops()
    replay_count = 0

    def policy(context, operation, *args, **kwargs):
        # The public selective cache consumes its entries during the first
        # replay. A retained graph can replay again by computing the same op.
        if operation in operations and (not context.is_recompute or replay_count == 1):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    forward, replay = create_selective_checkpoint_contexts(policy)

    class Replay:
        def __enter__(self):
            nonlocal replay_count
            replay_count += 1
            return replay.__enter__()

        def __exit__(self, *exc):
            return replay.__exit__(*exc)

    return forward, Replay()
