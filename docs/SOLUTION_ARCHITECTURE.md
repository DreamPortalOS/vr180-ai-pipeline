# System Solution Architecture — Two Delivery Routes

_Created 2026-06-24. Planning doc (per "先规划后开发"). Pairs with `ROADMAP.md` (sequencing) and `COMPETITOR_AND_BUSINESS.md` (market)._

## The strategic split

We now target **two distinct delivery formats** from the same source pipeline. They differ only in the
**render/projection stage** — everything before it (ingest, upscale, generate) is shared.

| | **Route 1 — 球幕影院 / Fulldome** | **Route 2 — VR 头显 / VR180** |
|---|---|---|
| Viewing | Dome projection, **no glasses** | VR headset (Quest / Vision Pro) |
| Stereo | **Monoscopic** (one image) | **Stereoscopic** (two eyes) |
| Projection | Domemaster — circular fisheye, azimuthal-equidistant, 180–220° | SBS equirectangular 180° + `sv3d`/`st3d` |
| Frame | Square, 4K² → 8K² | SBS, square-per-eye (2880²–4096²/eye) |
| Depth needed? | **No** (optional 2.5D parallax) | **Yes** (depth → disparity) |
| Hard problems | Projection accuracy, source resolution | + stereo fusion, ghosting, **nausea** |
| Risk / effort | **Low** — sharp & comfortable, ships fast | **High** — the part we're stuck on |
| Differentiation | Parity with buildvr.ai's fulldome | True-stereo + AI-native generation |

**Recommendation:** build **Route 1 (Fulldome) first.** It sidesteps the exact problems blocking us
(ghosting, eye-misalignment, nausea) because there is **no second eye** — every pixel serves one mono image,
so it's inherently sharper and comfortable. It matches the user's own Quest feedback ("comfortable, like a
curved screen, no 3D") and the competitor proves the market. Route 2 (VR180 stereo) continues in parallel as
the high-value differentiator, gated on the GPU stereo-quality models (DepthCrafter/StereoCrafter).

---

## Shared pipeline (both routes)

```
  ingest (2D clip)  ─┐
  or generate ───────┤→  [SeedVR2 upscale]  →  ┌──────────────────────────┐
  (prompt→Veo/Kling) │     (resolution)         │  RENDERER (pluggable)     │ →  encode + metadata  →  deliver
                     └──────────────────────────│  Route1: FulldomeRenderer │
                                                │  Route2: VR180Renderer    │
                                                └──────────────────────────┘
```

- **Ingest / Generate** — bring-your-own clip now; prompt→VideoGen later (archived `integrations/`).
- **SeedVR2 upscale** — `pipeline/video_upscaler.py` (board R-1). Shared, no-regret: both routes need resolution.
  Runs on the user's 4070S. See `SEEDVR2_SETUP.md`.
- **Renderer interface** — the one new abstraction. `render(frames) -> projected_frames`, two implementations.
- **Encode + metadata** — h265/MV-HEVC; fulldome needs no spherical metadata, VR180 needs `sv3d`/`st3d`.

---

## Route 1 — Fulldome (NEW build, recommended first)

**Format:** Domemaster = **circular fisheye, azimuthal-equidistant projection**, square frame, 180° (extendable
to 200°+). Center = zenith (straight up), circumference = horizon. Mono. 4K²/8K². (Refs: Loch Ness Productions
primer; Paul Bourke dome resources.)

**Pipeline:** upscaled 2D → **`pipeline/fulldome_mapper.py`** (ffmpeg `v360=input=flat:output=fisheye:h_fov=…:v_fov=…`,
mono) → square fisheye master → encode. **No depth, no stereo, no inpainting** → no ghosting, max sharpness.

**Build cost:** small — reuses the v360 machinery already in `equirectangular_mapper.py`, just a different output
projection and no stereo branch. This is the fastest path to a *clear, comfortable* deliverable.

**Optional 2.5D:** depth-driven subtle camera move / parallax for "pop" without true stereo — later, optional.

**Open questions (user):** target dome software/projector (affects exact FOV, tilt, resolution, and whether a
spherical-mirror warp is needed vs. a true fisheye lens). Playback test: any fisheye-aware player, or flat preview.

---

## Route 2 — VR180 stereo (continue, harder)

**Format:** what we have — SBS equirect 180°, square-per-eye, `sv3d`+`st3d` (Stereo Mode 2). Working & validated.

**Remaining quality work (all GPU, run on 4070S):**
- **Resolution:** SeedVR2 upscale (R-1) — shared.
- **Ghosting / eye-misalignment:** root cause is depth + disparity quality. → **DepthCrafter** (temporal-stable
  depth, kills shimmer) + **StereoCrafter** (clean disocclusion, no smear). These directly fix the "重影没对上".
- **"Boundary when looking up":** the source's limited vertical FOV → 180° generative **outpainting** (M1d), or
  accept a floor/ceiling. Stretching (current interim) trades sharpness.

**Why keep it:** true stereo + AI-native generation is the defensible differentiator vs buildvr.ai's mono.

---

## Competitor tech reverse (buildvr.ai) — informs both routes
See `COMPETITOR_AND_BUSINESS.md` §技术逆向 for detail. TL;DR: they are **depth-estimation + camera-reprojection**
(same family as us), productized into **6 output projections from one input** (360 mono, depth map, fisheye,
cubemap, **fulldome**, half-SBS) via a documented HEVC processing API. They lead with **mono** (depth/stereo is an
add-on) precisely to dodge nausea. **Takeaway:** our "one source → multiple projections" should be a first-class
design (the Renderer interface above), and fulldome is proven-shippable. Our edge stays: **true stereo + generation**.

---

## Build order
1. **SeedVR2 upscale** (R-1) — shared, unblocks resolution for both routes. *(deploying now on 4070S)*
2. **Route 1 Fulldome renderer** — small, high-quality, shippable. *(recommended next)*
3. **Route 2 stereo quality** — DepthCrafter + StereoCrafter on 4070S. *(parallel, harder)*
4. Refactor render stage behind the **Renderer interface** so both routes share ingest/upscale/encode.
5. **Outpainter (Stage 3.5)** — optional outpainting after equirectangular projection, before stereo rendering.
   Fills black zenith/nadir boundaries caused by limited source vertical FOV.
   Gradient mode (OpenCV, no model) ships first; AI mode (SDXL inpaint backend) is pluggable for later.

---

## Stage 3.5 — 180° Outpaint Fill (optional)

_Route 2 (VR180) only; no-op on Route 1 (fulldome)._

When a 2D source is projected to equirectangular (VR180 hemisphere), the top (zenith ≈ 20–30°) and bottom
(nadir) have **no source content** → pure black.  The outpainter fills these regions so the viewer sees a
continuous sky/ground instead of a hard black edge.

### Three modes & trade-offs

| Mode | Backend | Quality | Performance | Dependency |
|---|---|---|---|---|
| `none` | — | No fill | Zero cost | None |
| `gradient` | OpenCV (blur + edge extension) | Decent for uniform sky/ground; may show blur on complex textures | Real-time (2 ms/frame) | `opencv-python` |
| `ai` | Pluggable `AIOutpaintBackend` (SDXL inpaint / Seedance / custom) | Best — semantically fills with sky, clouds, terrain | Slow (1–5 s/frame on GPU) | `diffusers` + `torch` + deployed model |

### How it works (gradient mode)

1. `detect_black_boundary_mask()` scans top N% and bottom N% of the frame row-by-row, stopping at the
   first non-black row (threshold: mean pixel < `mask_threshold`).
2. `_gradient_outpaint_single()` copies the nearest valid row's content outward with a decreasing alpha,
   then applies a vertical Gaussian blur to blend the seam.
3. The original non-masked pixels are restored after blurring, so middle content is untouched.

### AI backend contract

```python
class AIOutpaintBackend(abc.ABC):
    def outpaint(self, frames: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]: ...
```

Implementations must handle the batch (multiple frames with the same mask).  A `MockAIOutpaintBackend` is
provided for testing.  For real deployment see `docs/OUTPAINT_SETUP.md` (placeholder — lead fills actual
model path + commands).

### Pipeline integration

```
equirectangular_mapper → [outpainter] → stereo_renderer → encode
                         ↑ optional
                   --outpaint {none,gradient,ai}
```

- `--outpaint gradient` is safe to use on **every** run — it detects black and only fills where needed.
- `--outpaint ai` requires a deployed backend; the CLI will error with a clear message if unavailable.
- Default is `none` so existing behavior is preserved.
