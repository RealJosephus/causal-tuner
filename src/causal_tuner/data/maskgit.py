"""Dense-only masked assistant image grids with same-position targets."""

import math
import re
import numpy as np
import torch
from .segments import canonical_segments, prediction_targets, runs


class MaskGIT:
    def __init__(self, tokenizer, probability, seed):
        self.tokenizer, self.probability, self.seed = tokenizer, probability, seed
        vocab = tokenizer.get_vocab()
        required = ("[MASK]", "<|im_start|>", "<|im_end|>")
        for token in required:
            if token not in vocab or tokenizer.encode(token) != [vocab[token]]:
                raise ValueError(f"MaskGIT requires a single-token {token!r}")
        self.mask, self.start, self.end = (vocab[t] for t in required)
        self.images = {value for key, value in vocab.items() if re.fullmatch(r"<\|image_[0-9]+\|>", key)}
        if not self.images:
            raise ValueError("MaskGIT requires image code tokens")

    def grids(self, ids, left, right):
        starts = [i for i in range(left, right) if ids[i] == self.start]
        for index, start in enumerate(starts):
            limit = starts[index+1] if index+1 < len(starts) else right
            ends = [i for i in range(start+1, limit) if ids[i] == self.end]
            if not ends:
                continue
            end = ends[0]
            header = self.tokenizer.decode(ids[start+1:min(end, start+33)])
            if not header.startswith(("assistant\n", "assistant\r\n")):
                continue
            images = [i for i in range(start+1, end) if ids[i] in self.images]
            groups, group, widths, row_width = [], [], [], 0
            for position in images:
                if group and position != group[-1]+1:
                    gap = self.tokenizer.decode(ids[group[-1]+1:position])
                    if gap in ("\n", "\r\n"):
                        widths.append(row_width)
                        row_width = 0
                    else:
                        groups.append((group, widths+[row_width]))
                        group, widths, row_width = [], [], 0
                group.append(position)
                row_width += 1
            if group:
                groups.append((group, widths+[row_width]))
            for group, widths in groups:
                if len(set(widths)) != 1:
                    raise ValueError("Assistant image grids must have equal row widths")
                yield group

    def __call__(self, batch, micro_step):
        if self.probability == 0:
            return batch
        if "prediction_targets" in batch or "attention_end" in batch:
            raise ValueError("Cannot apply MaskGIT twice to an already augmented prepared batch")
        output = dict(batch)
        tokens = batch["input_ids"].clone()
        targets, weights = prediction_targets(batch)
        end = torch.arange(tokens.shape[1]).expand_as(tokens).clone()
        segments = canonical_segments(batch["attention_mask"], batch["segment_ids"])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, micro_step]))
        for row, left, right in runs(segments):
            ids = batch["input_ids"][row].tolist()
            for grid in self.grids(ids, left, right):
                original_weights = batch["loss_weights"][row, grid]
                if not bool((original_weights > 0).any()) or rng.random() >= self.probability:
                    continue
                count = max(1, math.ceil(len(grid)*math.cos(math.pi*0.5*rng.random())))
                selected = torch.tensor(rng.choice(grid, count, replace=False).tolist())
                if not bool((batch["loss_weights"][row, selected] > 0).any()):
                    # Keep the sampled mask count, but replace one zero-weight
                    # selection with a useful target. This is bounded even for
                    # very sparse grids, unlike rejection sampling whole masks.
                    positive = [position for position, weight in zip(grid, original_weights.tolist()) if weight > 0]
                    selected[int(rng.integers(count))] = int(rng.choice(positive))
                first, last = grid[0], grid[-1]+1
                tokens[row, selected] = self.mask
                end[row, first:last] = last-1
                weights[row, max(left, first-1):last] = 0
                targets[row, selected] = batch["input_ids"][row, selected]
                weights[row, selected] = batch["loss_weights"][row, selected]
        output.update(input_ids=tokens, prediction_targets=targets, prediction_weights=weights, attention_end=end)
        return output
