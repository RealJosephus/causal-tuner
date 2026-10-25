# Causal Tuner

Causal Tuner provides single-device decoder training with LoRA and checkpoint tools.

## Installation

Training runs on `cuda:0` with compatible PyTorch, Triton, and FlashAttention 2 installations. Use `CUDA_VISIBLE_DEVICES` to select a GPU, and install the required CUDA backends for your platform. CPU execution is available for small-model development checks.

```bash
python -m pip install .
python -m pip install packaging ninja
python -m pip install 'flash-attn>=2.7,<3' --no-build-isolation
```

The package requires Python 3.11 or newer, PyTorch 2.8 or newer, and Triton 3.4 or newer. Triton is installed automatically on Linux; other platforms need a compatible backend installation. Use a CUDA-enabled PyTorch build compatible with the host driver. Install FlashAttention 2 against that PyTorch build; compilation from source requires the CUDA toolkit. See the [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention#installation-and-features).

The standard installation includes data processing, Hub access, W&B, and bitsandbytes. Datasets 5.1 is supported. Dependency ranges are listed in [pyproject.toml](pyproject.toml). The optional `cut` loss backend also requires `cut-cross-entropy`.

