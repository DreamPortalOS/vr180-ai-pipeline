# 验收清单 — 2026-06-26（更新：5 个 PR）

> 我（Claude）已 review + 本地验证全部代码。下面是**需要你亲自确认**的部分。打勾即可，有问题写后面发我。

---

## A. 五个 Cline PR（待你确认 + 合并）

合并顺序固定（stacked）：**#27 → #28 → #29 → #30 → #31**。

| PR | 内容 | CI | review 要点 |
|---|---|---|---|
| **#27** | SeedVR2 CLI 后端（修复版） | ✅ 绿 | 测试已修绿；12GB 参数齐（encode 分块+offload）；误提交文件已删；文档大小改成实测 3.16/0.47GB |
| **#28** | DepthCrafter 集成（骨架） | ✅ 绿 | `--depth-model` 默认仍 depth-anything；可插拔后端；mock 测试；setup 占位文档 |
| **#29** | R-4 等距批量提速（~10×） | ✅ 绿 | 每只眼整段一次 v360；输出仍方形 SBS + sv3d/st3d |
| **#30** | StereoCrafter（干净补遮挡） | ⚠️ 无 CI* | `--stereo-model` 默认仍 default；可插拔后端；mock 测试 |
| **#31** | 180° 外绘（gradient + AI） | ⚠️ 无 CI* | `--outpaint` 默认 none；gradient 开箱即用；AI 后端可插拔 |

> *#30/#31 没跑 CI——因为它们的 base 是上一个功能分支（不是 main），而 CI 只在"PR 到 main"时触发。
> **我已本地验证**：新模块 75 passed，全栈 312 passed，0 回归。合并进 main 后 CI 会自动跑。

**省事方案**：你回我 **"合"**，我就按序合并 + 处理所有 stacked/squash 冲突 + 把 #30/#31 retarget 到 main 让 CI 跑。

---

## B. Quest 实测（我发的两个真实成片）—— 判断下一步优先级的唯一依据

- [ ] **quest_googlegemini_vr180.mp4**（Skybox VR / DeoVR → 「180° 3D (SBS)」）
  - 重影对没对上（1-5）：____
  - 抬头/低头见黑边界？：____
  - 晕不晕？：____
  - 立体感？：____
  - 清晰度（SeedVR2 1.5×）：____
- [ ] **quest_googlegemini_fulldome.mp4**（鱼眼 / dome 播放器）
  - 鱼眼投影正不正？：____
  - 清晰度：____

---

## C. 需要你拍板

- [ ] **决策1**：同意我合并 #27~#31 吗？（我处理所有冲突 + retarget）→ 回 **"合"**
- [ ] **决策2**：哪个画质问题最严重？（重影 / 边界 / 晕 / 清晰度）→ 决定我先部署 DepthCrafter 还是 StereoCrafter 还是外绘
- [ ] **决策3**：阶段 G（接 AIGC 生成）是否启动？要的话给我一个引擎 API key

---

## D. 长期路线图
见同目录 `ROADMAP_LONGTERM.md`：阶段 Q(画质攻坚, 当前) → G(闭环生成) → P(工作流平台) → C(商业化)，外加 360 研究分支。

## 今早你只需 3 件事
1. 戴 Quest 给两个文件打分（B）。
2. 回我 **"合"** + **最严重的画质问题是哪个**（C 决策1+2）。
3. 看一眼长期路线图，有要调整的告诉我。
其余（合并、部署 DepthCrafter/StereoCrafter、跑新对比）我接。
