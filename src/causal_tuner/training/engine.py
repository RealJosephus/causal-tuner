"""Weighted gradient accumulation and deterministic optimizer scheduling."""

import math
import torch
from ..data.segments import prediction_targets, canonical_segments, runs
from ..kernels.loss import active_rows, projection_loss
from ..data.microbatch import repack_batches


def make_optimizer(model, args):
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": group, "weight_decay": args.weight_decay if decay else 0.0}
              for decay in (True, False) if (group := [p for p in parameters if (p.ndim > 1) == decay])]
    return torch.optim.AdamW(groups, lr=args.learning_rate,
                              betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon,
                              weight_decay=args.weight_decay, foreach=False, fused=all(p.is_cuda for p in parameters))


def learning_rate(args, step, total, warmup=None, decay_fraction=None):
    warmup = args.warmup_steps if warmup is None else warmup
    if total is not None:
        if total <= 0:
            raise ValueError("The training schedule must contain at least one update")
        if total == warmup == 1:
            warmup = 0
        elif not 0 <= warmup < total:
            raise ValueError("warmup_steps must be smaller than the training schedule horizon")
    if warmup and step < warmup:
        return args.learning_rate * step/warmup
    if args.scheduler == "constant" or (total is None and decay_fraction is None):
        return args.learning_rate
    fraction = min(1., max(0., (step-warmup)/(total-warmup) if decay_fraction is None else decay_fraction))
    factor = 1-fraction if args.scheduler == "linear" else (1+math.cos(math.pi*fraction))/2
    return args.learning_rate_end + factor*(args.learning_rate-args.learning_rate_end)


class Trainer:
    def __init__(self, model, optimizer, args, *, device):
        self.model, self.optimizer, self.args, self.device = model, optimizer, args, device

    def update(self, batches, *, step, total_steps, schedule=None):
        lr = learning_rate(self.args, *(schedule or (step, total_steps)))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_weight = torch.zeros_like(total_loss)
        microbatches = 0
        input_batches = 0
        microbatch_rows, execution = [], []
        valid_tokens = 0
        min_longest_segment = None
        def counted_batches():
            nonlocal input_batches
            for batch in batches:
                input_batches += 1
                yield batch

        for batch in repack_batches(counted_batches(), self.args.microbatch_token_budget,
                                    self.args.data_buffer_budget_mib*1024**2//4):
            microbatches += 1
            microbatch_rows.append(batch["input_ids"].numel())
            valid_tokens += int(batch["attention_mask"].sum())
            spans = batch.get("_spans")
            if spans is None:
                spans = runs(canonical_segments(batch["attention_mask"], batch.get("segment_ids")))
            longest = max((end-start for _, start, end in spans), default=0)
            min_longest_segment = longest if min_longest_segment is None else min(min_longest_segment, longest)
            targets, weights = prediction_targets(batch)
            rows = active_rows(weights, self.device)
            targets, weights = targets.to(self.device), weights.to(self.device)
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            hidden = self.model(batch)
            execution.append(dict(self.model.last_execution))
            loss = projection_loss(hidden, self.model.lm_head.weight, targets, weights,
                                   token_budget=self.args.loss_token_budget,
                                   compute_dtype=self.model.runtime.compute_dtype,
                                   implementation=self.args.loss_implementation,
                                   cut_chunk_size=self.args.cut_chunk_size,
                                   reuse_hidden=not self.model.model.norm.weight.requires_grad, rows=rows)
            # Loss backends retain whatever their backward needs. The fused
            # backend only retains gradients, so release the full model output.
            del hidden
            loss.backward()
            if self.model.offload is not None:
                self.model.offload.synchronize()
            total_loss.add_(loss.detach())
            total_weight.add_(weights.sum())
            del loss, batch
        if microbatches == 0:
            raise ValueError("An optimizer update requires at least one microbatch")
        denominator = total_weight.clamp_min(1.0)
        # The optimizer already owns the fixed trainable set, including restored state.
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        for param in parameters:
            if param.grad is None:
                raise RuntimeError("A trainable parameter has no gradient; the graph is disconnected")
        torch._foreach_div_([param.grad for param in parameters], denominator)
        if self.args.max_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm,
                                                  error_if_nonfinite=True, foreach=all(p.is_cuda for p in parameters))
        else:
            norm = torch.sqrt(sum(p.grad.float().norm().square() for p in parameters))
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        return {"loss": (total_loss/denominator).item(), "effective_weight": total_weight.item(),
                "grad_norm": norm.item(), "learning_rate": lr, "microbatches": microbatches,
                "input_batches": input_batches, "microbatch_rows": microbatch_rows, "execution": execution,
                "valid_tokens": valid_tokens, "min_longest_segment": min_longest_segment}
