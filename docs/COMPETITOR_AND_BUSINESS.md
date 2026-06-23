# 竞品逆向 & 商业模式规划：buildvr.ai

> 调研日期：2026-06-23
> 关联：`docs/STRATEGY_AI_VR180.md`

---

## 一、buildvr.ai 是什么

AI 驱动的「2D 视频 → VR」转换 SaaS。核心卖点：**不需要 360 相机，上传普通视频即可转成沉浸式 VR 内容**。

| 维度 | 内容 |
|------|------|
| 核心功能 | 2D 视频 → 360° 等距投影（AI depth mapping + spatial reconstruction） |
| 输出格式 | 6 种：360° monoscopic、depth map、HEVC/H.265、fisheye、fulldome、half-SBS |
| 配套工具 | **ScreenLab**（多环境预览：平面屏/弧形屏/穹顶）、VR 内容流媒体库 |
| 目标客户 | 影视工作室、广告 agency、纪录片团队、教育机构 |
| 输入支持 | MP4/AVI/MKV，最高 4K/8K |

---

## 二、关键逆向洞察（决定我们的差异化）

### 洞察 1：他们主打 monoscopic，刻意规避立体融合
buildvr 的主输出是 **360° 单目（monoscopic）**，depth map 只是可选附加项。
- **为什么**：双眼立体（true stereo）是最难、最易致晕的部分。单目 360° 只是"把平面贴到球面弧形屏"，沉浸但无真 3D，**几乎不会致晕**。
- **印证**：这正好对应用户 Quest 反馈——"拉远看舒适但没 3D，是个弧形 180° 屏幕"。那个舒适状态，本质就是 buildvr 的 monoscopic 体验。
- **启示**：我们做 **true stereo VR180** 是更高难度、更高价值的差异化，但必须解决致晕（立体精度）问题，否则不如直接做 monoscopic。

### 洞察 2：AI 转换额度极低 = 算力成本是核心成本
| 档位 | 价格 | AI VR 转换额度 | 分辨率 |
|------|------|---------------|--------|
| Explorer | $19/月 | **仅 1 分钟/月** | 4K |
| Pro Studio | $29/月 | **仅 2 分钟/月** | 4K |
| Enterprise | 联系销售 | 批量 + API | - |
- 1-2 分钟/月的额度低到离谱 → 说明每分钟转换的 GPU 成本很高。
- **启示**：商业模式必须把算力成本算清楚。定价应按"转换时长/分辨率"计量，而非无限订阅。我们的 Celery + 云 GPU worker 架构正是为此服务。

### 洞察 3：他们是"转换器"，不是"生成器"
buildvr 只做 video→VR 转换，用户要**自带视频**。
- **我们的独特机会**：我们覆盖 **从 creative/prompt → AI 生成 → VR180** 的全链路（依托 Kling/Seedance/Veo）。用户连原始视频都不用有，输入创意即可。这是 buildvr 做不到的。

---

## 三、我们的商业模式规划

### 定位
**「创意 → FPV VR180」全链路生成平台**（类 ComfyUI 的在线工具 + 一键成片）。
差异化三角：**真立体 3D** ×  **AI 原生生成（非仅转换）** × **FPV/特定垂直场景**。

### 三层产品
1. **Convert（对标 buildvr）**：用户上传 2D 视频 → VR180。守住基本盘。
2. **Generate（差异化）**：用户输入 prompt → 调 Kling/Seedance/Veo 生成 → 自动转 VR180。buildvr 没有。
3. **Studio（高端）**：工作流编排（类 ComfyUI 节点）、批量、API、企业定制。

### 定价模型（草案，按算力计量）
| 档 | 定位 | 计量 |
|----|------|------|
| Free | 引流 | 每月 N 秒预览档（baseline，低分辨率，带水印） |
| Creator | 个人 | 按分钟订阅 + 高质量档（StereoCrafter） |
| Studio | 团队 | API + 批量 + 8K + 无水印 |
| Enterprise | 定制 | 私有部署 / 专属 GPU |

### 成本结构关键
- 主成本 = GPU 算力（生成 + 立体转换 + 超分）。
- 控本手段：baseline 快速预览档（Mac/低端 GPU）兜底，高质量档（StereoCrafter，需 12-16GB GPU）按量计费。
- 我们的 4070S/3060 适合做 MVP 阶段的高质量档验证。

---

## 四、待定：是否注册 buildvr API 深入逆向
用户可注册其 API。若要深入，重点观察：
1. 转换 API 的参数（FOV、stereo/mono 切换、depth 强度）
2. 实际输出文件的 metadata（投影方式、stereo layout）
3. 处理时长与排队机制（推断其 GPU 调度）
4. 输出质量（与我们 baseline / StereoCrafter 档对比）

→ 建议在我们 baseline 跑通、Phase Q 启动前注册，用作质量基准对标。
