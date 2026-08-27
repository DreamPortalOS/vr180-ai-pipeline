# 长期开发路线图（2026-06-26）

## 现状：地基已搭好
- **转换管线**：2D → VR180(立体) + 球幕(单目)，真实片(googlegemini 10s) 端到端跑通。
- **清晰度**：SeedVR2 已本机部署 + 实测（2× 画质明显提升，~7s/帧）。
- **画质杠杆骨架**（Cline 已建代码 + mock 测试绿，真实模型待我部署）：
  - DepthCrafter（时序稳定深度 → 治重影根源）
  - StereoCrafter（干净补遮挡 → 治拖影/重影）
  - 180° 外绘（填上下黑边 → 治"抬头见边界"）
- **Prompt**：target-aware(180/360) 就绪（prompt_builder + prompt_lab）。
- **性能**：等距投影批量化（~10×）。

---

## 阶段 Q — 画质攻坚（当前，约 1-3 周）
**目标**：重影 / 边界 / 晕 / 清晰度逐个解到 Quest 可接受。
- [我] 本机部署 DepthCrafter + StereoCrafter 真实模型（同 SeedVR2 套路），12GB 调参。
- [我] 建立"最高画质"预设链：SeedVR2 2× → DepthCrafter → StereoCrafter → 180°外绘。
- [你+我] Quest 反馈闭环，迭代到三大问题解决。
- **风险**：12GB 串 4 个重模型会很慢/可能 OOM → 可能要降配，或确认是否上云 GPU。

## 阶段 G — 闭环生成（接 AIGC，约 2-4 周）
**目标**：从"自带片"升级到"应用内 文字→生成→转换"全链路。
- [Cline] 复活 `integrations/`（Kling/Seedance/Veo provider，archived 分支有底子）。
- [你] 提供一个引擎 API key。
- [你+我] 用 prompt_lab + Quest 反馈，把"标准 prompt 模板"固化进 prompt_builder。

## 阶段 P — 工作流平台（类 ComfyUI，约 1-2 月）
**目标**：非技术用户网页端完成全流程。
- 复活/重构 `web/` + PRD-v2：分镜 → 生成 → 转换 → 预览 → 导出，每步 checkpoint。
- 任务队列（重模型耗时长）+ 进度 + WebXR 预览。
- [Cline 做模块 · 我做架构]。
- **决策点**：自建 UI vs 做成 ComfyUI 自定义节点集（直接复用其生态/UI）。

## 阶段 C — 产品化/商业化（季度级）
- Convert / Generate / Studio 三层产品。
- **GPU 扩展**：本机 4070S 是开发盒；生产跑重模型链要云 GPU（4090/H100）。
- 计费、托管、用户系统（archived auth/db 有底子）。

## 横切：360 路线（研究，长期）
- 你提的 360 全向：需 360-native 生成 / 多视角拼接 / 大面积外绘。
- 策略：先把 180 两路打磨到位；360 作为研究分支推进。

---

## 近期 3 步（建议顺序）
1. **合并 #27~#31**（我来做，处理 stacked/squash 冲突）—— 你说"合"。
2. **我部署 DepthCrafter + StereoCrafter** → 跑 googlegemini 全链路 → 出新 Quest 对比。
3. 据 Quest 反馈定**阶段 G**（生成接入）的启动时机。
