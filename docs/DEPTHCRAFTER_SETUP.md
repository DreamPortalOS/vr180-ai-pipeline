# DepthCrafter Setup Guide

[Tencent/DepthCrafter](https://github.com/Tencent/DepthCrafter) provides
temporally-consistent video depth estimation.  This document describes how
to set up the DepthCrafter repository so the pipeline can use it.

> **⚠️ Status: Skeleton**
> The real deployment steps (exact model names, download commands, inference
> CLI call) will be filled in by the lead once the DepthCrafter repo has been
> tested end-to-end on the target GPU.  The sections below document the
> intended layout so that ``DepthCrafterEstimator`` can locate the artifacts.

---

## Repository Layout

Clone the DepthCrafter repo **outside** this project:

```bash
git clone https://github.com/Tencent/DepthCrafter.git /path/to/DepthCrafter
```

Inside the clone, the inference entry point is expected at one of:

- `run.py`
- `inference.py`
- `scripts/inference.py`
- `depthcrafter/inference.py`

> TODO(lead): Decide which file serves as the CLI entry point and update
> `_find_inference_script()` in `pipeline/depth_crafter.py` if needed.

---

## Checkpoints / Models

Model files go into `<repo_dir>/checkpoints/` (configurable via
`--checkpoint_dir` or `DEPTHCRAFTER_CKPT_DIR`).

| File | Size | Source |
|------|------|--------|
| TODO | TODO | TODO |

> TODO(lead): Fill in the actual checkpoint file name, size, and HuggingFace
> / other download URL.

---

## Inference Command (Reference)

The intended CLI invocation (once the repo is ready):

```bash
python run.py \
    --video /path/to/input.mp4 \
    --output_dir /path/to/output \
    --max_resolution 1024 \
    --checkpoint_dir /path/to/checkpoints
```

> TODO(lead): Verify that the command flags match the actual inference script.
> Update `CLIBackend.estimate_video()` in `pipeline/depth_crafter.py` if the
> flags differ.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEPTHCRAFTER_REPO_DIR` | *(required)* | Path to cloned DepthCrafter repo |
| `DEPTHCRAFTER_PYTHON` | `python` | Python executable for inference |
| `DEPTHCRAFTER_CKPT_DIR` | `<repo_dir>/checkpoints` | Model checkpoint directory |
| `DEPTHCRAFTER_MAX_RES` | `1024` | Max resolution (short side) for inference |

---

## CLI Flags (pipeline)

When using `scripts/run_pipeline.py`:

```
--depth-model {depth-anything,depthcrafter}
                    Depth estimation backend (default: depth-anything)
--depthcrafter-repo-dir DIR
                    DepthCrafter repository path (or env DEPTHCRAFTER_REPO_DIR)
--depthcrafter-python EXE
                    Python for DepthCrafter inference (or env DEPTHCRAFTER_PYTHON)
--depthcrafter-checkpoint-dir DIR
                    DepthCrafter checkpoint dir (or env DEPTHCRAFTER_CKPT_DIR)
--depthcrafter-max-res N
                    Max resolution for DepthCrafter (or env DEPTHCRAFTER_MAX_RES)
```

---

## Testing Without a GPU

All unit tests in `tests/test_depth_crafter.py` use mocks — they never call
the real model or require CUDA.  Run them with:

```bash
pytest tests/test_depth_crafter.py -v
