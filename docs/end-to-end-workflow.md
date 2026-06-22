# End-to-End Workflow — 2D FPV → VR180 飞行体验

## 完整制作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       制作流程总览                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🖥️ Google Gemini Omni (Platform)                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. 打开 gemini.google.com                              │   │
│  │  2. 选择 Text to Video → Gemini Omni Flash               │   │
│  │  3. 粘贴 FPV 飞行提示词                                   │   │
│  │  4. 生成 10秒 16:9 1080p 视频                            │   │
│  │  5. 下载 MP4                                             │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  📦 vr180-ai-pipeline (Local)                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  python scripts/run_pipeline.py -i input.mp4 -o output  │   │
│  │                                                         │   │
│  │  Stage 1: Depth Estimation (Depth Anything V2)          │   │
│  │  Stage 2: Stereo Disparity (Left/Right Eye View)        │   │
│  │  Stage 3: Equirectangular Projection (3840×1920 SBS)    │   │
│  │  Stage 4: VR Metadata Embedding (YouTube VR Ready)      │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  🥽 VR Headset / YouTube VR                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Quest: Skybox VR Player  /  USB 传文件                  │   │
│  │  Vision Pro: Moon Player / AirDrop                       │   │
│  │  YouTube VR: 直接上传，自动识别                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 步骤详解

### 步骤 1: 生成 2D FPV 视频

**操作平台**: Google Gemini Omni (gemini.google.com)
**所需**: Google AI Plus/Pro/Ultra 订阅

1. 打开 Gemini → 选择 Omni 模型
2. 选择 **Text to Video** 模式
3. **16:9** 宽高比，10秒时长
4. 粘贴以下测试提示词：

```
Cinematic FPV drone flying through a narrow red rock canyon,
first-person perspective weaving between towering sandstone walls,
morning golden hour light, warm earthy tones,
16:9 landscape, smooth flying, clear foreground-background separation.
```

5. 生成并等待 1-2 分钟
6. 审查视频质量
7. 下载 MP4 文件

### 步骤 2: 运行转换流水线

```bash
# 进入项目目录
cd ~/vr180-ai-pipeline

# 激活虚拟环境
source .venv/bin/activate

# 运行完整流水线
python scripts/run_pipeline.py -i ~/Downloads/gemini_omni_fpv.mp4 -o ~/Downloads/fpv_vr180.mp4 --fps 30
```

**预期输出:**
- 第 1 阶段（深度估计）: ~2-10 秒/帧（取决于模型大小和设备）
- 第 2 阶段（立体渲染）: ~0.1 秒/帧
- 第 3 阶段（等距柱状投影）: ~0.5 秒/帧
- 第 4 阶段（编码）: ~5-10 秒

**在 Apple M2 Max 上（mps 加速），10秒视频约需 2-5 分钟**

### 步骤 3: 验证 VR180 文件

检查元数据是否嵌入正确：

```bash
ffprobe -v quiet -print_format json -show_streams ~/Downloads/fpv_vr180.mp4 | grep -A5 -i "spherical\|stereo"
```

### 步骤 4: 在 VR 中播放

**Meta Quest:**
1. 连接 USB 线
2. 复制文件到 `Quest/内部共享存储空间/Download/`
3. 打开 Skybox VR Player → 选择视频 → 设置为 VR180 SBS

**YouTube VR:**
1. 上传到 YouTube（设置为 "Unlisted" 用于测试）
2. YouTube 自动识别 Spherical Video V2 元数据
3. 在 YouTube VR 应用或桌面浏览器中查看（拖拽观看）

---

## 故障排除

### 深度估计不准确

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 深度图模糊 | 使用 Small 模型 | 改用 `--model-size base` |
| 帧间闪烁 | 无时域一致性 | 使用 Video Depth Anything 替代 |
| 前景/背景混淆 | 场景太复杂 | 优化提示词, 增加前景物体 |

### 立体效果不理想

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无立体感 | 视差太小 | 增大 `--max-disparity 0.08` |
| 眼睛疲劳 | 视差过大 | 减小 `--max-disparity 0.03` |
| 边缘空洞 | 深度过渡太陡 | 降低 `--max-disparity` |

### 视频不被识别为 VR180

| 问题 | 解决方案 |
|------|----------|
| YouTube 显示为 2D | 确保使用 `--codec h264`（H.265 兼容性差） |
| Quest 不识别 | 使用 Skybox VR Player 手动选择格式 |
| 画面扭曲 | 检查 `--src-hfov` 是否匹配原视频 FOV |

---

## 质量阶梯

```
                    第一阶段（快速验证）
                    ═══════════════════
Model:    Depth Anything V2 Small  (24.8M params)
Device:   mps (Apple Silicon)
Output:   3840×1920 SBS, H.264, CRF 23
Time:     ~1-2 min for 10s video

                    第二阶段（最佳质量）
                    ═══════════════════
Model:    Depth Anything V2 Large  (335M params)
Device:   cuda (NVIDIA GPU)
Output:   7680×3840 SBS, H.265, CRF 18
Time:     ~5-10 min for 10s video

                    第三阶段（未来）
                    ═══════════════════
Model:    Video Depth Anything (CVPR 2025)
Device:   cuda
Output:   7680×3840 SBS, H.265 10-bit, CRF 16
Features: Temporal-consistent depth, optical flow
```