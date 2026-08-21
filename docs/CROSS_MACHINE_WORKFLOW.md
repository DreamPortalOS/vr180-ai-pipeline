# 跨机分段管线操作指引（Windows CUDA ↔ Mac MPS 接力）

> 对应 issue #36（V-3）。重模型分工：**SeedVR2 超分在 Windows 4070S（CUDA-only）**，
> **DepthCrafter / StereoCrafter 在 Mac M2 Max（96GB 统一内存，MPS）**。
> 本指引描述"Windows 跑一半 → 人工拷贝 → Mac 跑重模型 → 拷回 → 编码"的三步接力流程。
> 文件传输是**人工**的（U 盘 / scp / 网盘均可），管线本身不做任何网络同步。

## 概念：job manifest

每次分段运行由一份 **job manifest**（JSON）串起来：

```json
{
  "version": 1,
  "job_id": "myvideo-a1b2c3d4",
  "source": "/path/to/myvideo.mp4",
  "source_hash": "<sha256>",
  "stages": [
    {
      "name": "depth",
      "status": "done",
      "machine": "mac-mps",
      "inputs": [],
      "outputs": ["…/myvideo_vr180_temp/depth"],
      "params": {"depth_model": "depthcrafter", "model_size": "small"},
      "hashes": {"<file>": "<sha256>"}
    }
  ]
}
```

五个规范 stage 名：`upscale` → `depth` → `stereo` → `project` → `encode`
（`project` = 等距投影 + outpaint；`encode` = VR 元数据 + 编码）。

- `--stages a,b,c`：只跑指定子集（缺省 = 全部，行为与旧版完全一致）。
- `--manifest <path>`：跑完把每个 stage 的产物 hash 写进 manifest。
- `--resume-from <path>`：续跑。**先校验**已完成 stage 的产物 hash 与源文件 hash，
  任何一个不匹配都会明确报错并以退出码 1 终止——不会静默用坏数据接着跑。
- `--machine <label>`：写入 manifest 的机器标签（缺省自动生成，如 `windows-cuda` /
  `darwin-mps`；建议显式传 `win-cuda` / `mac-mps`）。

⚠️ 分段模式强制走**批处理路径**（`--quality standard/high` 默认的流式路径不可拆分，
加了 `--stages` / `--resume-from` 会自动退回批处理路径）。

---

## 第一步：Windows（CUDA）— SeedVR2 超分

```powershell
python scripts/run_pipeline.py `
    --input video/myvideo.mp4 `
    --video-upscale seedvr2 --video-upscale-factor 2 `
    --stages upscale `
    --manifest job_myvideo.json `
    --machine win-cuda `
    --temp-dir work/myvideo_temp
```

产物：
- `work/myvideo_temp/myvideo_seedvr2_2x.mp4`（超分后的视频）
- `job_myvideo.json`（`upscale` 标为 done，含产物 hash）

## 拷贝到 Mac

把这三样拷到 Mac（保持相对路径一致最省事）：

1. `job_myvideo.json`
2. `work/myvideo_temp/` 整个目录（含超分视频）
3. 源文件 `video/myvideo.mp4`（manifest 记录了它的 hash，续跑时要校验）

## 第二步：Mac（MPS）— DepthCrafter + StereoCrafter

```bash
# 注意 --input 指向超分后的视频（它就是本机的"源"）
python scripts/run_pipeline.py \
    --input work/myvideo_temp/myvideo_seedvr2_2x.mp4 \
    --depth-model depthcrafter --stereo-model stereocrafter \
    --device mps \
    --stages depth,stereo \
    --manifest job_myvideo.json \
    --machine mac-mps \
    --temp-dir work/myvideo_temp
```

产物写入 `work/myvideo_temp/depth/`、`left/`、`right/`，manifest 追加
`depth` / `stereo` 两条 done 记录（含产物目录与关键参数）。

> 若 manifest 的 `source_hash` 是在 Windows 上对**原始**视频算的，而 Mac 上
> `--input` 用的是超分视频，可用 `--resume-from job_myvideo.json` 代替
> `--manifest`——resume 校验的是 `--input` 指向的文件，请保证它与 manifest
> 的 `source` 一致；跨机接力时每台机器各自用 `--manifest` 顺序更新同一份
> JSON 即可（已完成 stage 不会被重跑，只要不在 `--stages` 里点名）。

## 拷回 Windows

把 `job_myvideo.json` + `work/myvideo_temp/`（现在含 depth/left/right 帧）拷回 Windows。

## 第三步：Windows — 投影 + 编码

```powershell
python scripts/run_pipeline.py `
    --input work/myvideo_temp/myvideo_seedvr2_2x.mp4 `
    --stages project,encode `
    --manifest job_myvideo.json `
    --machine win-cuda `
    --temp-dir work/myvideo_temp `
    --output myvideo_vr180.mp4
```

- `project`：从 `myvideo_temp/left|right/` 读帧 → 等距投影 SBS →（可选）outpaint。
- `encode`：从 `myvideo_temp/equirect/` 读帧 → H.264/H.265 编码 + `sv3d`/`st3d` 元数据。

完成后 `job_myvideo.json` 五个 stage 全部 done，即整条 job 的完整审计记录
（谁在哪台机器、用什么参数、产物 hash 是多少）。

## 断点续跑（同一台机器崩溃恢复）

```bash
python scripts/run_pipeline.py \
    --input work/myvideo_temp/myvideo_seedvr2_2x.mp4 \
    --resume-from job_myvideo.json \
    --manifest job_myvideo.json \
    --temp-dir work/myvideo_temp
```

已完成 stage 逐一做 hash 校验后被跳过；hash 不匹配 → 打印期望/实际 hash 并退出
（退出码 1），此时应重跑对应 stage 而不是强行续跑。

## 常见问题

- **hash mismatch 报错**：产物在拷贝后被改动/损坏，或 `--input` 指错了文件。
  重新拷贝或重跑该 stage。
- **流式路径不能用 `--stages`**：流式管线是单趟融合处理，无法分段；
  分段接力请用批处理路径（加 `--stages` 即自动切换）。
- **fulldome 投影不支持分段**：`--projection fulldome` 与 manifest 参数互斥。
