# VR180 AI Pipeline — Execution Roadmap

_Last updated: 2026-06-24. Owner tags: **[cline]** local autonomous dev · **[GPU]** needs remote GPU · **[lead]** orchestration/integration/review · **[user]** decision/credentials needed._

This is the **execution** plan (what to build, in what order, with done-criteria).
`docs/PRD-v2-vr180-studio.md` is the product vision; the live tactical queue is now
**GitHub Issues** — task cards executed via Claude Code CLI; see `CLAUDE.md` and `docs/DEV_PROCESS.md`.

---

## 2026-08-21 状态快照与新冲刺（Quality Sprint v3）

**已落地**（PR #1–#32 全部合并）：M0 稳定化 ✅ · M1a 几何/舒适默认值 ✅（#22）· SeedVR2 集成 ✅（#26/#27，4070S 已部署）·
fulldome 渲染器 ✅（#17）· DepthCrafter/StereoCrafter 可插拔封装 ✅（#28，真实推理待 Mac 部署）· 批处理等距 ~10× ✅（#29）·
180° 外绘 ✅（#31）· 生成层 Kling/Seedance/Veo ✅（#32）。Prompt 工程已砍掉（实测 Veo 对提示词不敏感）。

**Quest 实测结论（2026-06-27，全维度 1 分）→ 两个根因**：
1. **清晰度**：每眼仅 1920px（为省内存压的），180° 铺开像素密度极低。管线自带流式 O(1) 内存模式可出 3840–7680/眼 → **做成默认高清路径**。
2. **重影/立体弱**：视差 0.02 太保守 + 单帧深度噪声大 → 短期做**参数扫描 A/B 工具**（视差/汇聚面），长期靠 Mac (M2 Max) 跑 DepthCrafter/StereoCrafter 出干净深度与补遮挡。

**新冲刺任务卡（GitHub Issues，V-1 … V-6）**：V-1 高清流式默认路径（P0）· V-2 立体参数扫描工具（P0）·
V-3 跨机分段管线 + job manifest（P1）· V-4 长片分块内存管理（P1）· V-5 VR180 输出 QA 校验器（P2）· V-6 fulldome 定位文档（P2）。

---

## North Star
Turn a text prompt or a flat 2D clip into an **immersive** video, end-to-end.
As of 2026-06-24 this splits into **two delivery routes** off one shared pipeline — see
**`docs/SOLUTION_ARCHITECTURE.md`**:
- **Route 1 — Fulldome / 球幕** (mono, no glasses): circular-fisheye domemaster. *Recommended first — sharp & comfortable.*
- **Route 2 — VR180** (stereo, headset): square-per-eye SBS equirect + `sv3d`/`st3d`. *Differentiator, harder.*
Shared front-end: ingest/generate → **SeedVR2 upscale** → pluggable renderer → encode.

---

## M0 — Stabilize · [cline] ✅ DONE (2026-06-24)
- ✅ **T-A** PR #8 (T2 auth) — added `passlib[bcrypt]` (+pinned `bcrypt<5`); CI green.
- ✅ **T-B** test failures — found already-passing on main; only the metadata test remained.
- ✅ **T-C** PR #10 VR180 metadata — root cause was `inject_via_spatialmedia_cli` calling bare `python3`
  (resolved to a Python without `spatialmedia`); fixed to `sys.executable`. `sv3d`/`st3d` now injected; CI green.

**Merge order to consolidate:** #9 Hermes → #8 auth → #10 metadata (all green; board file overlaps, so merge serially and resolve the board once).

---

## M1 — Output Quality · [lead] + [cline] + [GPU]
The pipeline is structurally correct (stereo + VR metadata) but the *picture* isn't immersive yet.
User test feedback (2026-06-24): black borders / FOV not filled, eyes not aligned, low effective
resolution, visible seams. Tackle **geometry & comfort first, then sharpness, then GPU-heavy quality.**

### M1a — Geometry & comfort (no GPU) · [lead → cline to make default]
Researched + validated interactively 2026-06-24 (delivered `googlegemini_vr180_v2.mp4`); needs to become
the pipeline DEFAULT, not ad-hoc CLI flags.
1. **Per-eye must be SQUARE (1:1).** 180°×180° → square half-equirect. The old `3840×1920` (2:1) vertically
   squished content to half-height — the core "looks wrong / doesn't fill" bug. Use square per-eye
   (validated `2880×2880` → SBS 5760×2880).
2. **Fill the FOV.** `--src-hfov 150` fills ~99% of the hemisphere (vs the tiny window at 90°). Tradeoff is
   spherical stretch at edges — acceptable interim until M1d outpainting. Make a sensible default + document.
3. **Comfortable stereo.** Dropped `max_disparity` 0.05 → 0.02 so the eyes fuse (fixes "not aligned").
   Tune convergence so the main subject sits at the zero-parallax plane.
- [ ] [cline] bake these as defaults in `run_pipeline.py` / `EquirectangularMapper` / `StereoRenderer`; add a test.

### M1b — Resolution / sharpness (the "pixel upgrade") · [cline build] + [user GPU run]  ← ACTIVE (board R-1)
Target per-eye **2560²–3840²** (Quest 3 panel 2064×2208/eye, Quest 2 1832×1920). Source here is only 720p,
so mapping straight to 2880² is soft — the dominant cause of "completely unclear".
**Decision (2026-06-24): upscale the SOURCE first with SeedVR2** (ByteDance, ICLR2026 — 2026 SOTA video SR,
temporal-aware, best on short AI-generated compressed clips). Runs on the user's RTX 4070S/3060 (12GB; 3B model
≥8GB, GGUF for less). `batch_size` must be `4n+1` for temporal consistency. This supersedes the old per-frame
Real-ESRGAN in `pipeline/upscaler.py` (fast but flickers, "enhanced" not "reconstructed"). Bump bitrate for hi-res.
Root-cause alternative (M2): generate the source natively at 1080p–4K instead of 720p. Pick canonical output res by
target device (Quest vs Vision Pro).

### M1c — Temporal depth + stereo fidelity · [GPU]
DepthCrafter (temporal-stable depth, stops shimmer) + StereoCrafter (clean disocclusions). Current
single-frame Depth-Anything + disparity-shift renderer stays as the no-GPU fallback.

### M1d — 180° generative outpainting (biggest long-term visual win) · [GPU]
Replace the "stretch to fill" interim (M1a.2) with generative outpainting that extends real content to a
true 180° — no distortion, no black corners. `pipeline/research/ai_outpainter.py` is a stub to build on.

### M1e — Equirect performance · [cline]
Mapper spawns **2 ffmpeg subprocesses per frame** (~1.5–3s/frame). Move to a single batched `v360` pass
or in-process mapping to cut conversion ~10×.

**M1 done when:** a clip fills the headset FOV, eyes fuse comfortably, looks sharp at Quest PPD, no seams.

## M1.5 — Audio / soundtrack (PLAN ONLY for now, do not build yet) · [lead] decision
Output currently has **no audio**. Decide the path before building:
- **Option A — synchronous generation:** request audio/music together with the source clip if the VideoGen
  provider supports it (e.g. Veo audio), or generate a matched track from the same prompt at generation time.
  Pro: semantically matched to scene; Con: provider-dependent, less control.
- **Option B — post stage:** add an audio stage after conversion — AI music/SFX (e.g. text-to-audio model)
  or a licensed ambient library keyed off the prompt/scene tags, muxed into the final MP4 via ffmpeg.
  Pro: decoupled, swappable, controllable length/loop; Con: less tightly synced to on-screen action.
- **Leaning:** Option B (post stage) as the default pipeline capability, with Option A as an opt-in when the
  provider returns audio. Revisit after M1 basics land. **No development now.**

---

## M2 — End-to-end vertical slice (the demo) · [lead] + [cline] + [user]
Wire the existing parts into one flow: **prompt → generate → convert → notify**.

1. `prompt_builder` → `integrations` VideoGen provider (Veo/Kling/Seedance) to produce the source clip. **[user]** API keys.
2. Source clip → `run_pipeline` VR180 conversion (M0/M1 output).
3. On completion → Hermes (`notifications/feishu.py`) fires the card. **[user]** Feishu webhook URL.
4. Persist job + result via `db/` (T1); gate write endpoints with auth (T2/PR #8); enforce quota.
5. Expose as one web API endpoint: `POST /jobs {prompt}` → job id → status → download URL.
   **Gap to close:** the **streaming pipeline path injects no VR metadata** — unify it with the spherical injector
   so every output route produces valid VR180.

**M2 done when:** one API call turns a prompt into a downloadable VR180 file and pings Feishu.

---

## M3 — Studio frontend (Phase C) · [lead] + [cline]
ComfyUI-style web app over the M2 API: prompt → generate → convert → in-browser preview → download.
Three tiers from `PRD-v2`: **Convert** (bring your own clip) · **Generate** (prompt→VR180) · **Studio** (params, batch, history).
**M3 done when:** a non-technical user completes prompt→preview→download in the browser, no CLI.

---

## M4 — Commercialization · [user] + [lead]
Pricing tiers, API key management + metering (build on T2 auth + quota), billing, usage dashboard.
**M4 done when:** a new user can sign up, get a key, run within quota, and be billed.

---

## Cross-cutting / tech debt (fold into the sprint that touches the area)
- VR metadata must be identical across CLI, streaming, and API output paths (single source of truth).
- Decide the canonical target device(s) — Quest (SBS equirect) vs Apple Vision Pro (MV-HEVC via `spatial_converter`). **[user]**
- CI installs the spatial-media git dep and runs the metadata regression test.
- **Workflow:** cline and lead share one working tree → branch/stash collisions. Run cline and lead orchestration
  in separate git worktrees (or time-slice) to stop clobbering each other's uncommitted work.
- Keep `.clinerules` discipline: serial dependent tasks, branch from latest `main`, never `--no-verify`.
