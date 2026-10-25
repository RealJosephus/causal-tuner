# Causal Tuner

Causal Tuner provides single-device LoRA training, dataset preparation, checkpoint continuation, and weight export for a supported MQA decoder architecture. It integrates with Hugging Face Hub for data and checkpoint storage, and with Weights & Biases for optional run tracking.

Model loading reads dimensions, context length, and RoPE settings from the model's configuration and requires the supported architecture defined in [config.py](src/causal_tuner/modeling/config.py) and standard safetensors weights. Python code from model repositories is not imported.

## Installation

Training runs on `cuda:0` with compatible PyTorch, Triton, and FlashAttention 2 installations. Use `CUDA_VISIBLE_DEVICES` to select a GPU, and install the required CUDA backends for your platform. CPU execution is available for small-model development checks.

```bash
python -m pip install .
python -m pip install packaging ninja
python -m pip install 'flash-attn>=2.7,<3' --no-build-isolation
```

The package requires Python 3.11 or newer, PyTorch 2.8 or newer, and Triton 3.4 or newer. Triton is installed automatically on Linux; other platforms need a compatible backend installation. Use a CUDA-enabled PyTorch build compatible with the host driver. Install FlashAttention 2 against that PyTorch build; compilation from source requires the CUDA toolkit. See the [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).

The standard installation includes data processing, Hub access, W&B, and bitsandbytes. Datasets 5.1 is supported. Dependency ranges are listed in [pyproject.toml](pyproject.toml). The optional `cut` loss backend also requires `cut-cross-entropy`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 causal-tune \
  --repo-id /models/base-model \
  --dataset-name /data/train.jsonl \
  --max-sequence-length 4096 --num-train-epochs 3 \
  --save-optimizer-state --save-steps 10 \
  --output-dir causal-tuner-output
```

Paths, lengths, and step counts in the examples are placeholders for the intended run. `--repo-id` accepts a local model directory or a Hub model repository. `--dataset-name` accepts a local data source or a Hub dataset repository.

`--num-train-epochs` sets an epoch-based run. `--max-steps` defaults to `-1`; setting it to a positive value selects an optimizer-update horizon instead and takes priority over epochs. Streaming epoch-based runs require an integer epoch count.

| Command | Purpose |
|---|---|
| `causal-tune` | Train or continue from a checkpoint. |
| `causal-tune-prepare` | Prepare data for later training. |
| `causal-tune-export` | Export weights from a saved checkpoint. |
| `causal-tune-merge` | Merge an exported LoRA adapter into standalone model weights. |

Use `--help` with each command for its options. Training and preprocessing defaults are defined in [arguments.json](src/causal_tuner/arguments.json), with constraints in [arguments.py](src/causal_tuner/arguments.py). Dashed and underscored option names are equivalent. Boolean options accept a bare switch, an explicit value, or `--no-...`.

### Training configuration

The default configuration uses rank-256 LoRA on the seven decoder projections, BF16 base weights and computation, and FP32 trainable weights, gradients, and Adam state. LoRA uses unit scaling.

`--lora-train-embed` and `--lora-train-lm-head` independently enable embedding and output-table training. `--quantization nf4` applies only to frozen decoder projections. `--param-dtype` controls frozen base storage. `--precision-recipe reference_bf16` changes trainable weights, gradients, and optimizer state to BF16. Full-parameter AdamW training is also available, subject to persistent-state memory checks.

Gradient accumulation sums weighted negative log-likelihood and gradients across an optimizer update, then divides once by `max(total_effective_weight, 1)`. AdamW applies weight decay to matrices. Padding-token lookup suppresses the configured padding row's gradient.

Warmup starts at zero and counts completed optimizer updates. It must fit within a known update horizon; a one-update run skips warmup. `--learning-rate-end` is the absolute final learning rate for a decay schedule.

## Data

### Raw records

Local raw data can be JSONL, a JSON array, CSV, TSV, or Parquet. Remote datasets are loaded through Hugging Face Datasets. Raw text receives no additional EOS token.

For chat records, use `--dataset-text-field messages`. Assistant-only and last-assistant-only loss modes require generation spans in the chat template. Assistant `loss` and `weight` values can be non-negative floats. Token weighting uses the last overlapping generation span.

Chat templates use a bounded compilation cache. The `tojson` filter defaults to `ensure_ascii=False` and accepts explicit `ensure_ascii`, `indent`, `separators`, and `sort_keys` options. CSV and TSV readers allow long fields within the data budget, including quoted multiline records, and restore temporary field limits after each read.

EndPrompt changes logical positions without expanding the token sequence to the logical length. Packing inserts no separator between records. Attention boundaries and prediction targets remain separate across records, repeated segment labels, and masked gaps.

`--maskgit-probability` enables same-position supervision for assistant image tokens with explicit bidirectional attention groups. A selected image grid retains at least one positive-weight target without increasing its sampled mask count.

### Prepared datasets

```bash
causal-tune-prepare --repo-id /models/base-model \
  --dataset-name /data/train.jsonl --output-dir /data/prepared

causal-tune --repo-id /models/base-model \
  --dataset-name /data/prepared --dataset-format prepared \
  --max-steps 100 --save-optimizer-state --save-steps 10
```

Preparation uses `Dataset.from_generator` and removes columns whose values can be reconstructed exactly for every record. Input IDs are stored once; default masks, labels, segment IDs, positions, and unit weights are reconstructed during training. Non-default fields retain their values. Stored fields are variable-length lists without added padding: IDs and positions use int32, masks use int8, and weights retain float64 precision until batch construction. Local output uses `Dataset.save_to_disk` and can be read with `datasets.load_from_disk`.

Generation and saving use bounded row batches and a temporary build cache. Completed output is committed atomically. A `manifest.json` file records preprocessing and tokenizer signatures and checksums of the native files. Training requires a supported prepared format with matching metadata.

Reuse the same preprocessing options during training. Templates, EndPrompt, and packing are not applied a second time. Epoch planning uses the prepared record count without iterating token batches. Local prepared datasets use memory mapping.

### Batches and execution packs

`--total-batch-size` counts input records or stored packed rows per logical batch. `--gradient-accumulation-steps` combines logical batches into one optimizer update. `--max-sequence-length` limits each input or packed row, rather than the combined token count of a logical batch.

Each logical batch compacts valid tokens into one tensor row while preserving positions, attention boundaries, and prediction targets. Padding, internal gaps, and empty tail rows are excluded from projection and MLP computation.

Before device transfer, `--microbatch-token-budget` groups independent attention spans into execution packs. Its default is 12288 tokens. Compatible short packs can be combined across accumulated logical batches within the same optimizer update. Each pack is one forward/backward microbatch. A span larger than the target runs intact on its own. Zero preserves the original grouping.

This grouping preserves optimizer-update membership, supervision weights, MaskGIT sampling, data cursors, and the learning-rate schedule. It does not require preparing the dataset again. `--packing` separately controls how short raw records form logical input packs.

### Buffering and streaming

Tokenization, shuffle, packing, batch construction, and synchronous CPU prefetch have byte limits. Records and batches that exceed their stage budget raise an error. Batch loss weights use FP32. Transient Hub failures use capped-backoff retries; reopened streams verify the delivered prefix before continuing. Permanent failures stop the operation.

Epoch-based streaming reads to EOF without a counting pass and requires an integer number of epochs. Source counts are committed after successful updates. When a source count is known, it defines a fixed decay horizon of 105% of the estimated total, with progress capped at the estimate per epoch. After warmup, decay starts at the source position committed by the last warmup update. Unknown counts use update-based warmup followed by a constant learning rate. A positive `--max-steps` sets the update horizon instead. Continuation retains the saved plan and cursor.

## Execution and memory

CUDA attention uses FlashAttention 2 with native MQA and packed-sequence layouts. Projections and MLPs use LoRA backward operators, SwiGLU, and bounded workspaces. The fused projection loss processes supervised rows in tiles. RMSNorm, RoPE, and NF4 decoding use CUDA kernels. CPU and `vanilla` attention paths provide small-model reference implementations; CUDA execution does not silently fall back to them.

`--gradient-checkpointing auto` selects activation retention, caching, and checkpointing from the model configuration and the tokens in each execution pack. The automatic cache policy applies to the default architecture with rank-256 LoRA, frozen tables, unquantized BF16 base/compute, FP32 trainable state, FlashAttention 2, and single-layer checkpoint groups. Other configurations use checkpointing. The selection rules are defined in [execution.py](src/causal_tuner/training/execution.py).

Explicit `enabled` always checkpoints, while `disabled` retains activations. `--remat-group-size` adds an outer checkpoint over a group of layers. `--remat-inner-checkpoint False` retains intermediate activations during group replay instead of adding inner layer checkpoints. Groups of size one are unchanged.

| Option | Behavior |
|---|---|
| `--mlp-chunk-size` | Token rows per MLP tile. Unset selects from the execution pack; zero disables outer tiling. |
| `--projection-chunk-size` | Token rows per projection tile. Zero uses an untiled base projection with bounded adapter and backward workspaces. |
| `--mlp-cache-budget-gib` | Budget for saved LoRA MLP intermediates on non-checkpointed layers. Unset uses the automatic policy; zero disables caching. |
| `--attention-cache-budget-gib` | Budget for saved FlashAttention 2 outputs in checkpointed layers. Unset uses the automatic policy; zero disables caching. Explicit positive values require single-layer FlashAttention 2 checkpointing. |
| `--activation-offload` | `auto` makes bounded CPU storage available to checkpointed layers; `none` disables it; `cpu` enables it explicitly. |

Automatic cache budgets are capped using available device memory and estimates for other activations, workspaces, gradients, and Adam state. Cache budgets cover saved intermediates, not total GPU allocation or temporary workspaces. Execution choices are captured during the forward pass for checkpoint replay. Saved intermediates remain immutable during backward. A repeated backward on a retained graph recomputes cached attention outputs.

Tile sizes can change BF16 rounding and FP32 gradient-reduction order. Frozen BF16 pre-normalization can reconstruct normalized inputs from saved outputs when weight magnitudes are within the supported range, which also changes backward rounding. Other normalization paths use input-based derivatives.

`--loss-token-budget` limits projected rows per fused-loss tile. Rows without supervision skip projection. The optional `--loss-implementation cut` backend requires a frozen output head and uses its own tiling and dtype-dependent gradient filtering. `--cut-chunk-size` controls its outer chunks.

`--gpu-memory-budget-gib 0` lets CUDA manage available GPU memory. A positive value adds an explicit memory limit and an attention-workspace guard. `--host-memory-budget-gib`, `--checkpoint-host-budget-gib`, and `--pinned-memory-budget-gib` bound host storage and activation offload. Checkpoint and pinned budgets must fit within the host budget. Offload uses pageable storage, a worker, and bounded pinned staging buffers. Exceeding a configured limit raises an error.

## Save and resume

`--save-steps` controls periodic checkpoints and defaults to zero. `--save-optimizer-state` includes Adam state and defaults to false. Enable optimizer-state saving when continuation is required; Hub training checkpoints require it. `--save-total-limit` controls the number of retained local checkpoint directories.

Use `--stop-after-updates` to save and stop after a specified number of updates in the current invocation while preserving the training schedule. Repeating the training command loads the latest compatible checkpoint and continues the saved plan. Continuation requires matching dependencies, configuration, model, and dataset signatures. Missing optimizer state is rejected rather than recreated.

| File | Contents |
|---|---|
| `state.safetensors` | Trainable weights, saved Adam tensors and scalar steps, and Torch RNG state. |
| `manifest.json` | Checksums, Python/NumPy RNG state, data cursor, schedule, tracking identity, and compatibility metadata. |

Local checkpoints commit atomically. Tensor serialization uses bounded staging independently of file size, with a default staging limit of 64 MiB. Checkpoint state remains in one standard safetensors file.

## Hugging Face Hub

Authenticate with `hf auth login` or `HF_TOKEN` when required. To prepare data on the Hub and train with Hub checkpoints:

```bash
causal-tune-prepare --repo-id /models/base-model \
  --dataset-name source-owner/raw-data --dataset-split train \
  --dataset-text-field text --dataset-streaming \
  --max-sequence-length 4096 --packing \
  --hub-dataset-repo-id account/prepared-data

CUDA_VISIBLE_DEVICES=0 causal-tune --repo-id /models/base-model \
  --dataset-name account/prepared-data --dataset-format prepared \
  --dataset-text-field text --dataset-streaming \
  --max-sequence-length 4096 --packing \
  --max-steps 1000 --save-steps 50 --save-optimizer-state \
  --hub-model-repo-id account/training-run
```

Output repositories default to public visibility with manually approved gated access: `--hub-private False --hub-gated manual`. The repository page is public, while gated downloads require an approved user token. `--hub-private` and `--hub-gated auto|off` change these settings. Access settings are applied and verified before uploads. No output repository is created unless an output repository option is supplied.

Hub preparation uses a temporary native Dataset cache, calls `Dataset.push_to_hub`, and uploads the preprocessing manifest. The library manages Parquet compression and file sharding. It does not retain a prepared copy in `output_dir`. Serialization row counts are bounded using the largest prepared record and the data-stage budget, independently of file sharding.

Hub prepared data uses `datasets.load_dataset`; `--dataset-streaming` selects its iterable loader. Data and manifest use one resolved commit per invocation. Model weights, configuration, and the default tokenizer also share one commit. A separate processor source can be selected explicitly.

`--revision` and `--dataset-revision` select source revisions. When omitted, each invocation resolves the current default-branch commit. Continuation still checks that the resolved inputs match the saved identities.

### Hub checkpoints

`--hub-model-repo-id` stores checkpoints under `checkpoint/` in a model repository. Each save creates one Git commit. The Hub client reads a seekable safetensors stream directly from stable tensors, pre-uploads it, and commits it with the manifest. Training pauses during transfer to keep tensors at one optimizer step. A complete local checkpoint copy is not required for upload.

Restore pins one commit, downloads and verifies the state file into temporary disk storage, and copies tensor slices with bounded staging. Temporary disk space must accommodate the state file. The temporary file is released after restore.

Earlier checkpoints remain in repository history. Local `--save-total-limit` does not prune Hub history, and deleting a current path does not reclaim its historical storage. Final exports are published at the repository root in a separate commit alongside `checkpoint/`.

## Export weights

Training completion exports the configured adapter or model and tokenizer files. An incomplete or completed checkpoint can also be exported independently:

```bash
causal-tune-export --checkpoint causal-tuner-output/checkpoint-00000010 \
  --output-dir adapter-step-10 --format adapter

causal-tune-export --checkpoint account/training-run \
  --checkpoint-revision COMMIT_SHA \
  --output-dir merged-step-10 --format model --device cuda:0
```

Export writes a new local directory. Model weights are written to one `model.safetensors` file, with tensors merged individually and serialized in bounded chunks. It reads Hub repositories without changing their settings or creating commits. Omitting `--checkpoint-revision` resolves the current default branch. Export defaults to the saved output format and CPU loading; NF4 export requires a suitable CUDA device. `--format model` can merge a LoRA checkpoint into base weights. NF4 merges are identified as dequantized-base merges.

Saved base and processor revisions are reused and their identities checked. `--repo-id`, `--revision`, `--processor-repo-id`, and `--processor-revision` allow assets to be relocated while preserving those identities.

Independent export does not load training data, allocate an optimizer, restore training RNG, start W&B, or modify the source checkpoint. Exports commit atomically and include tokenizer files. Final export in Hub mode uses temporary disk space, which is released after upload. A failed final export can be retried from a completed checkpoint without loading optimizer state. Repeating an already completed and exported run exits without further updates.

## Merge an adapter

Use `causal-tune-merge` when the adapter has already been exported and a training checkpoint is not available:

```bash
causal-tune-merge --adapter causal-tuner-output/adapter-00000010 \
  --output-dir merged-model

causal-tune-merge --adapter account/adapter --adapter-revision COMMIT_SHA \
  --repo-id account/base-model --revision BASE_COMMIT_SHA \
  --output-dir merged-model --device cuda:0 --dtype bfloat16
```

From an uninstalled source checkout, set `PYTHONPATH` to `src` and run `python -m causal_tuner.merge` with the same options.

`--adapter` accepts a local directory or Hub repository containing `adapter_config.json` and `adapter_model.safetensors` exported by Causal Tuner. The base model defaults to `base_model_name_or_path` in the adapter configuration; `--repo-id` overrides its location. Use the same base weights and revision used for training. Remote adapter and base revisions are resolved to immutable commits before download. Model architecture and adapter tensor keys and shapes are checked before loading base weights.

The command merges `base + (lora_alpha / r) * B @ A`, restores any saved embedding and output tables, and writes one `model.safetensors` file, model configuration and tokenizer files to a new local directory. No sharding option is needed: weights are merged individually and written in bounded chunks without an extra whole-model CPU copy. Tokenizer files come from the adapter when it includes `tokenizer.json`, otherwise from the base model. Existing output directories are rejected, and source files are left unchanged. The output is committed atomically and is not uploaded to the Hub.

Loading and merging default to CPU. `--dtype` controls base storage and output precision; its default comes from the base configuration, falling back to BF16. Merge arithmetic uses FP32. The command loads the base model and adapter without training data or optimizer state, reusing the host memory budget checks used by checkpoint export. It supports the seven-projection LoRA format exported by this project, including optional trained embedding and output tables; other adapter variants are rejected. Merging always uses unquantized base weights, including for NF4-trained adapters. To merge against the dequantized NF4 base instead, use `causal-tune-export --format model --device cuda:0` with the training checkpoint.

## Tracking

The terminal shows completed optimizer steps, percentage, loss, step duration, and estimated remaining training time after every update. ETA uses the most recent 20 updates after the invocation's first warmup update. It excludes Triton-reported compilation and disk-cache loading, model loading, checkpoint saving, and logging. Remaining time is unavailable until a timing sample exists, or when streaming has no known update total; sequence lengths can change the estimate. The timing window restarts on continuation while the completed step count is restored.

Enable W&B with `--use-wandb`, after `wandb login` or setting `WANDB_API_KEY`. `--wandb-project`, `--wandb-run-name`, and optional `--wandb-entity` select the project, display name, and user or team.

The run ID, project, and entity are stored in local and Hub checkpoints. Continuation uses that identity through the SDK resume API. Changing project or entity for an existing tracked checkpoint is rejected.

At `--logging-steps` intervals and completion, JSON output and W&B record loss, learning rate, gradient norm, token counts, tokens per second, update duration, and data progress. They also include `progress_percent`, `eta_seconds`, `training_seconds`, `training_elapsed_seconds`, `mean_step_seconds`, and `compilation_seconds`. `--track-memory` adds GPU/host memory and offload metrics. Raw `elapsed_seconds` includes compilation within the update; `training_seconds` subtracts the compiler's reported time. Both exclude checkpoint saves, logging, and lookahead performed outside the timed update.

Charts use optimizer `step` as a custom axis, separate from W&B's append-only event counter. Replaying updates after an interruption can produce repeated optimizer-step values. Checkpoints remain in the selected local or Hub destination. Failed training finishes the W&B invocation with a nonzero exit status.

## Development checks

The test suite includes numerical and gradient comparisons, tensor-storage checks, data processing, checkpoint replay, offload, export, and continuation. Backend probes run before model weights are allocated. Optional-backend tests can be skipped when their dependencies are unavailable.

In a development environment with the required dependencies installed:

```bash
python -m pip install pytest httpx
python -m pytest -q -ra
python -m causal_tuner.smoke --device cpu
python -m causal_tuner.smoke --device cuda --offload
python -m causal_tuner.smoke --device cuda --bf16 --offload
```

Smoke commands use small model shapes to exercise training, saving, and continuation. Their inputs and scope are defined in [smoke.py](src/causal_tuner/smoke.py).
