# autoresearch-spark

Autonomous LLM pretraining research on the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) (GB10 / Blackwell).

Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for the GB10 single-GPU platform, with [OpenShell](https://github.com/NVIDIA/OpenShell) integration for sandboxed agent execution.

![progress](progress.png)

*118 experiments ran autonomously overnight on a DGX Spark. The agent reduced val_bpb from 1.8794 (baseline) to 1.2265 -- a 34.7% improvement -- by adapting hyperparameters to the platform's constraints. See [LEARNINGS.md](LEARNINGS.md) for details.*

## What is this?

Give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and a better model.

The training code is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea: you don't touch the Python files like a normal researcher. Instead, you program the `program.md` Markdown file that provides context to the AI agent and sets up your autonomous research org. The agent edits `train.py`, runs experiments, and iterates.

This fork targets the DGX Spark specifically. The GB10 GPU (Blackwell, compute capability 12.1 / sm_121a) has unique toolchain requirements and much lower throughput than an H100, so the hyperparameter sweet spots are different. The agent discovers these differences automatically.

## How it works

The repo has three files that matter:

- **`prepare.py`** -- fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** -- the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, model size, etc.
- **`program.md`** -- baseline instructions for the agent. Point your agent here and let it go.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) -- lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start (bare metal)

**Requirements:** DGX Spark (or any single NVIDIA GPU), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

Then point your AI agent at `program.md` and let it run.

## Running with OpenShell (sandboxed)

[OpenShell](https://github.com/NVIDIA/OpenShell) provides a sandboxed container environment for autonomous agents. The agent runs with minimal permissions -- it can't install arbitrary packages, can't access the network beyond what's explicitly allowed, and can't modify anything outside its sandbox. This is important when you're letting an AI run unsupervised for hours.

### How the sandboxing works

1. **The container image is built ahead of time** with all dependencies pre-installed (`sandbox/Dockerfile`). PyTorch, the tokenizer libraries, and agent tools (Claude, OpenCode, Codex) are all baked into the image.

2. **At runtime, the agent gets a locked-down environment.** The OpenShell policy files (`sandbox/policy.yaml`, `sandbox/policy-dev.yaml`) define exactly what the agent can access:
   - **Filesystem:** read-only access to system paths, read-write only to `/sandbox` and `/tmp`
   - **Network:** only the specific endpoints needed (Anthropic API, GitHub for git, HuggingFace for data download, NVIDIA inference API)
   - **Process:** runs as an unprivileged `sandbox` user, not root
   - **Landlock:** Linux security module further restricts filesystem access

3. **The agent can modify code and run experiments** but can't escape the sandbox, install backdoors, exfiltrate data, or do anything unexpected. If the agent tries something malicious or buggy, the damage is contained.

### Quick start with OpenShell

```bash
# Build the sandbox image
docker build -f sandbox/Dockerfile -t autoresearch-spark .

# Launch with OpenShell
openshell sandbox create \
  --remote my-spark \
  --gpu \
  --provider claude \
  --provider github \
  --from autoresearch-spark \
  -- claude
```

This launches an autoresearch sandbox on your DGX Spark with Claude as the autonomous researcher.

### Why CUDA 13.0 + cu128?

The GB10 GPU is compute capability 12.1 (sm_121a), which creates a unique toolchain situation:

- **CUDA 13.0 devel base image:** provides `ptxas` with sm_121a support, which Triton needs to compile optimized GPU kernels.
- **PyTorch cu128 wheels:** the cu130 wheels are not yet functional on this platform, but cu128 works correctly on the CUDA 13.0 runtime (backward compatible).
- **`TRITON_PTXAS_PATH`** is set globally so Triton finds the CUDA 13.0 ptxas automatically.

## GB10 platform notes

The GB10 is significantly slower than an H100 for this workload (MFU ~0.86% vs ~40%). The agent compensates by discovering that smaller batch sizes are critical -- the default `TOTAL_BATCH_SIZE=2^19` only yields ~31 training steps in 5 minutes on GB10, vs ~950 on H100. Reducing to `2^14` gives ~689 steps and dramatically better results.

Key adaptations the agent discovered (from ~135 experiments):

| Change | Impact |
|--------|--------|
| `TOTAL_BATCH_SIZE` 2^19 -> 2^14 | val_bpb 1.88 -> 1.31 |
| `DEPTH` 8 -> 4 | More steps per time budget |
| `WINDOW_PATTERN` "SSSL" -> "L" | Full-context beats sliding window |
| `WEIGHT_DECAY` 0.2 -> 0.0 | No weight decay helps at small batch |
| `ADAM_BETAS` (0.8, 0.95) -> (0.9, 0.99) | Higher beta2 for small batches |
| `HEAD_DIM` 128 -> 96 | 4 heads at model_dim=384 |
| Slow Muon momentum warmup (5000 steps) | Gradual warmup improves stability |

See [LEARNINGS.md](LEARNINGS.md) for the full details and [docs/spark-gb10.md](docs/spark-gb10.md) for the GB10-specific compatibility fixes.

## GPU auto-detection

MFU (model FLOPS utilization) is reported relative to the detected GPU's peak BF16 FLOPS. The lookup table in `train.py` includes:

| GPU | Peak BF16 TFLOPS |
|-----|------------------|
| H100/H200 | 989.5 |
| A100 | 312.0 |
| B200 | 2250.0 |
| GB10 (DGX Spark) | 125.0 |
| MI300X/MI308X/MI325X | 1307.4 |
| MI250X | 383.0 |

Unknown GPUs fall back to H100 with a warning.

## Project structure

```
prepare.py           -- constants, data prep + runtime utilities (do not modify)
train.py             -- model, optimizer, training loop (agent modifies this)
generate.py          -- interactive inference from a trained checkpoint
program.md           -- agent instructions
LEARNINGS.md         -- platform-specific findings from experiments
pyproject.toml       -- dependencies
analysis.ipynb       -- notebook for analyzing results.tsv
docs/spark-gb10.md   -- GB10 compatibility details (SDPA, Triton ptxas)
sandbox/
  Dockerfile         -- OpenShell sandbox image (CUDA 13.0, PyTorch cu128)
  policy.yaml        -- runtime network/filesystem policy (locked down)
  policy-dev.yaml    -- development policy (adds PyPI access)
```

## spark-appendix branch

The [spark-appendix](../../tree/spark-appendix) branch adds extra tooling:

- `benchmark_flops.py` -- empirical BF16 TFLOPS measurement for your GPU

## Upstream

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Notable forks of the upstream:

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (macOS / MPS)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD ROCm / MI300X)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (macOS / MLX)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
