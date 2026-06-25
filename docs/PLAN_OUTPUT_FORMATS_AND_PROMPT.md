# 输出格式规划：180 飞行影院 vs 360 球幕，以及与 Prompt 的联动

> 创建：2026-06-25 · 规划文档（先规划后开发）。配套 `SOLUTION_ARCHITECTURE.md`（两路线）、
> `PROMPT_GUIDE_VR180.md`（prompt 原则）、`CLINE_TASK_BOARD.md`（执行看板）。
> 触发：用户提出「VR180 ≠ VR360，输出不一定是球幕；180=飞行影院，360=球幕影厅；这更多取决于
> 让 AI 生成 180 还是 360 的素材」。本文把这个判断落成可执行规划。

---

## 0. 结论先行（TL;DR）

- **「输出 180 还是 360」本质由素材的覆盖范围决定，而素材覆盖范围由生成 prompt 决定** —— 用户判断成立。
- 把两个常被混淆的维度拆开：
  - **覆盖维度（Coverage）**：前向 ~180°（飞行影院）vs 全向 360°（球幕影厅）。
  - **立体维度（Stereo）**：单目 mono（投影，不戴眼镜）vs 双目 stereo（VR 头显）。
  - 现仓库的「Route 1 Fulldome / Route 2 VR180」其实是这两维里的两个点；用户说的「180 vs 360」是**覆盖维度**。
- **素材现实**：主流 AIGC 视频工具（Veo / Kling / Seedance / Wan）只产出**平面 2D 矩形视频**，不产出原生 360 全景。所以：
  - **180（前向）今天就能做**：平面源 → 投影到前向半球。**同一平面源**出两种成片：
    ① 180 鱼眼球幕（mono，飞行影院） ② VR180 SBS（stereo，头显）。
  - **360（全向）今天难**：缺的 75% 球面要靠 360 原生生成 / 多视角拼接 / 大面积外绘，三者都更难、质量更低 → 研究路线。
- **因此 prompt 工程要按输出目标分流（target-aware）**，而不是只有一套 VR180 模板。

---

## 1. 三种交付目标与管线映射

| 目标 | 覆盖 | 立体 | 观看设备 | 管线投影 | 源 prompt 意图 | 现状 |
|---|---|---|---|---|---|---|
| **VR180 头显** | 前向 180° | stereo | Quest / Vision Pro | flat→depth→stereo→equirect180 SBS + `sv3d`/`st3d` | 前向广角 FPV、慢速、地平线稳 | ✅ 已有 |
| **180 鱼眼球幕 / 飞行影院** | 前向 180° 半球 | mono | 弧幕 / 半球幕投影 | flat→fisheye180（`v360` 单 pass） | 前向广角 FPV、强景深分层 | 🔜 R-5（已派 Cline，见看板 DISPATCH-1） |
| **360 球幕 / 全向** | 360°×180° | mono 或 stereo | 全球幕 / 头显 360 | 需要 equirect360 源 | 360 / 全景 或 环绕覆盖 | 🔬 研究（P1–P2） |

> 术语提醒：天文馆式「球幕（dome）」物理上通常是 **180° 半球**，用鱼眼 domemaster 投影；
> 「360 全向」指观众可**任意环顾**的全景。两者不要混。用户语境里：
> 「飞行影院」≈ 前向沉浸（180），「球幕影厅」≈ 全向沉浸（360）。

---

## 2. 素材来源的硬约束（为什么 180 可达、360 难）

- AIGC 出的是普通相机 FOV（~50–90°，prompt 可拉更广）的**平面片**。
- 把平面片投到**前向半球（180）** = 中心填满、边缘渐隐或 AI 补绘 —— 当前管线 + `fulldome_mapper`（R-5）正是干这个。**可达。**
- 要真 **360**，缺的球面必须来自：
  1. **360 原生生成** —— 少数全景扩散 / 360 视频模型；FPV 画质的主流工具基本不支持。
  2. **多视角生成 + 拼接** —— 同场景多次 yaw 旋转视角，再 stitch 成 equirect。
  3. **平面 + 时序外绘** —— 仓库 `pipeline/research/temporal_outpainter.py`、`ai_outpainter.py` 已有原型。
- 结论：**P0 把 180 的两种成片做扎实；360 列为 P1–P2 研究项。**

---

## 3. Prompt 工程方案（target-aware）

现状：`pipeline/prompt_builder.py` 的 `wrap_prompt_for_vr180(user_prompt, scene_type)` 只有一套 VR180 约束
（`scene_type ∈ {fpv, walkthrough, orbit, static}`），追加正向 + 负向 prompt。**已本地实测正常**（fpv/orbit/static 输出已验证）。

提案：升级为**按输出目标分流**。

- 建议签名：`wrap_prompt(user_prompt, scene_type="fpv", target="vr180_flight")`，
  `target ∈ {vr180_flight, fulldome_180, vr360_dome}`（旧函数保留为兼容 alias）。
- 各 target 差异：

| target | 立体 | 推荐 FOV 措辞 | 运动约束 | 负向 prompt |
|---|---|---|---|---|
| `vr180_flight`（头显立体） | stereo | wide cinematic ~120° | 最严（防 vergence 致晕：慢速、稳地平线、三层景深） | 严格（含边缘视差、rushing past edges） |
| `fulldome_180`（飞行影院 mono） | mono | wider ~150–180° look | 稳地平线（坐姿飞行防晕），但无双眼冲突 → 可更广更动 | 可放宽边缘视差类负向 |
| `vr360_dome`（全向） | mono/stereo | 全景/环绕覆盖 | 取决于后端能力 | 若主流模型 → 明确标注「需 360-capable 后端」 |

- 关键点：**单目（mono）没有双眼视差冲突**，所以飞行影院可以比 VR180 头显更广、运动更自由；
  VR180 头显的抗晕约束必须最严。一套模板套两种目标会两头不讨好。

---

## 4. 联调（Prompt × 生成 × 管线）测试计划

分两层，**第二层需要一个 AIGC 引擎的 API key**：

### 4.1 离线层（现在就能做，无需 key）
- 新增 `scripts/prompt_lab.py`：对同一创意，按 `{target × scene_type}` 批量产出 prompt 变体 + 一个 `manifest.json`，便于对比 / 版本管理 / 交接。
- 当前 `wrap_prompt_for_vr180` 已本地跑通；离线层只依赖它，不依赖任何外部 API。

### 4.2 真·联调层（需 key）
- 选 1 个引擎：**Kling**（运动控制最强）/ **Seedance**（国内可用）/ **Veo**（画质最好）。
- 用 prompt_lab 的变体各生成 5–8s 短片 → 跑对应管线（180 鱼眼 / VR180 SBS）→ Quest / 投影实测 → 回填「哪个变体最稳 / 最沉浸」。
- 这是真正能 tweak prompt 的闭环。**没有 key，只能做到离线层。**

---

## 5. 交付任务拆分（可交给 agent / Co-work / Cline）

| 编号 | 任务 | 归属 | 依赖 |
|---|---|---|---|
| **P-1** | `prompt_builder` target-aware 重构：加 `target` 参数 + `fulldome_180` / `vr360_dome` 模板 + 测试（旧函数保留为 alias） | Cline 可做（纯代码、CI 可绿） | 无 |
| **P-2** | `scripts/prompt_lab.py` 离线变体工具 + `manifest.json` + 测试 | Cline 可做 | P-1 |
| **P-3** | 真·联调：接 1 个引擎，按目标各生成短片跑管线，实测回填 | 需 key + 真人/lead | P-1, P-2, **API key** |
| **P-4** | 360 源可行性 spike：调研 360 原生生成 / 多视角拼接 / 时序外绘，选一条路 | lead / 研究 | 无 |
| **P-5** | 文档：PRD + ROADMAP 编码两种输出格式 | lead（本规划即第一版） | 无 |

> 注：R-5（fulldome_mapper，看板 DISPATCH-1）是 180 鱼眼球幕成片的**管线侧**前置，已派给 Cline；
> 本文 P-1/P-2 是 prompt 侧前置。两条线可并行。

---

## 6. 需要用户拍板的开放问题

1. **引擎 + key**：真·联调用哪个引擎？能否提供一个 API key（Kling / Seedance / Veo 任一）？
   没有 key 则先把离线层（P-1 / P-2）做掉。
2. **球幕硬件**：飞行影院 / 球幕的具体设备（投影仪 / 鱼眼镜头 / 弧幕半径 / `dome_fov` / 是否需要球面镜 warp），
   直接影响 fulldome 的 FOV / tilt / 分辨率参数。
3. **360 优先级**：360 全向是否纳入近期路线，还是先把 180 的两种成片打磨到位再说。
