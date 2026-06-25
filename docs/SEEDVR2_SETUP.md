# SeedVR2 Source Upscaling — Deployment on RTX 4070S (Windows)

_Goal: upscale the **720p source clip** to ~2880p **before** VR180/fulldome conversion. The 720p source is the
dominant cause of "completely unclear". SeedVR2 (ByteDance, ICLR2026) is the 2026 SOTA temporal video super-res —
best-in-class on short, AI-generated, compressed clips (exactly our input)._

> **Platform note:** the official `ByteDance-Seed/SeedVR` repo targets Linux/H100 and depends on `flash_attn` +
> NVIDIA `apex`, which are painful to build on Windows. **On Windows, use Path A (ComfyUI).** Use Path B only on
> Linux/WSL2 for headless automation.

---

## Path A — ComfyUI-SeedVR2 (recommended on Windows/4070S)

The ComfyUI node is built for consumer GPUs (block-swap, GGUF quant, tiling). SeedVR2 v2.5 runs even the 7B model
on 8GB; the **3B** model is comfortable on the 4070S's 12GB.

1. **Install ComfyUI** (portable Windows build) — https://github.com/comfyanonymous/ComfyUI
2. **Install the node** (ComfyUI-Manager → search "SeedVR2", or git clone into `ComfyUI/custom_nodes/`):
   `https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler`
3. **Download the 3B model** into the node's models folder:
   - `seedvr2_ema_3b_fp16.safetensors` (best quality, fits 12GB), or
   - `seedvr2_ema_3b-Q4_K_M.gguf` (lower VRAM / faster) — from HF `ByteDance-Seed/SeedVR2-3B` (or the node's HF mirror `numz/SeedVR2_comfyUI`).
4. **Workflow:** Load Video → SeedVR2 Upscaler → Save Video. Key settings for 12GB:
   - **Model:** 3B fp16 (or GGUF if VRAM-tight)
   - **Target resolution:** short side **1440** (→ 1280×720 becomes ~2560×1440) or push to **2160** if VRAM allows
   - **`batch_size` MUST be `4n+1`** (1, 5, 9, 13, 17 …) — this is how SeedVR2 keeps frames temporally consistent.
     Start at **5**; raise until VRAM is ~80% full, lower if OOM.
   - **Block swap:** raise it if you OOM (trades speed for VRAM); 0 if you have headroom.
5. **Run on our clip:** drop `video/googlegemini.mp4` in, export → `googlegemini_2560.mp4`.

---

## Path B — Headless (Linux / WSL2 only, for automation)

```bash
git clone https://github.com/bytedance-seed/SeedVR.git && cd SeedVR
conda create -n seedvr python=3.10 -y && conda activate seedvr
pip install -r requirements.txt
pip install flash_attn==2.5.9.post1 --no-build-isolation   # needs CUDA toolchain
# apex from a prebuilt wheel matching your CUDA/torch

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ByteDance-Seed/SeedVR2-3B", local_dir="ckpts/")
PY

# inference takes a FOLDER OF FRAMES, not an mp4:
ffmpeg -i input.mp4 frames_in/%06d.png
python projects/inference_seedvr2_3b.py \
  --video_path frames_in --output_dir frames_out \
  --res_h 1440 --res_w 2560 --sp_size 1 --seed 0
ffmpeg -framerate 24 -i frames_out/%06d.png -c:v libx265 -crf 16 -pix_fmt yuv420p googlegemini_2560.mp4
```
VRAM: a single 12GB card handles short clips at ~1440p; reduce `res_*` / clip length if OOM (the repo's reference
"1×H100-80G = 100×720×1280" is far above 12GB, so keep sequences short and resolution modest).

---

## Then convert (unchanged pipeline)

The upscaled clip is just a higher-res input — feed it straight into the existing converter:
```bash
PYTHONPATH=. python scripts/run_pipeline.py \
  --input video/googlegemini_2560.mp4 \
  --output video/googlegemini_vr180_v3.mp4 \
  --src-hfov 150 --max-disparity 0.02 --codec h265 --crf 16
# fulldome variant arrives with pipeline/fulldome_mapper.py (Route 1, board)
```

**Acceptance:** the `_v3` output should be visibly sharper than v2 — that confirms source resolution was the bottleneck.
Send it back for a headset test. (cline task R-1 will wrap this as a `--video-upscale` flag once the manual path is proven.)
