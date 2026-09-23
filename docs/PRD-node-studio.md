# Immersive Node Studio — 产品需求文档 (PRD)

> **2026-09-22 决定**：MiniMax 整条线已删除（owner：只到 2K，不能到 4K）。下文凡提到 MiniMax 的供应商/里程碑均作废，视频节点只保留 mock | Seedance。


> **项目名（暂定）**: Immersive Node Studio / 沉浸式节点工作室
> **版本**: v0.1 DRAFT
> **日期**: 2026-09-17
> **状态**: 待 owner 确认后推进实现
> **上位文档**: `docs/SOLUTION_ARCHITECTURE.md` · `docs/PRD-v2-vr180-studio.md`（平台层已归档参考）· `docs/prompt-design-guide.md`

---

## 0. 一句话

在网页画布上用**节点**串起「脚本 → 生图 → 生视频 → 转换 → 3D 预览 → 导出」，
同一工程同时产出 **VR180** 与 **球幕** 成片；生成可走本地 LiteLLM / SenseNova U1（免费档试用），
也可一键切 MiniMax / Seedance 付费。

---

## 1. 背景与动机

### 1.1 当前痛点

| 痛点 | 证据 |
|---|---|
| 改一版 prompt / 参数要 CLI 重跑，反馈周期长 | VR180 素材曾花十轮 prompt 才合规 |
| 几何问题在花钱渲染后才发现 | 球幕「内容只铺到 r≈0.6」是在 50 元生成 + 35 分钟渲染后才测出 |
| 脚本、图、视频、转换、预览分散在文件与脚本里 | `GEN_PROMPT_*.txt` / `scripts/generate.py` / `run_pipeline.py` / 独立预览器 |
| 生成供应商切换成本高 | Seedance 已接线；MiniMax 已实现后被 revert；本地/litellm 未接 |
| 缺少「所见即所得」的球面/立体检查 | dome-preview（PR #376）只覆盖球幕覆盖度，未进创作主流程 |

### 1.2 目标用户

| 角色 | 核心诉求 |
|---|---|
| **Owner / 内容导演** | 改脚本、试构图、控成本，不碰 CLI |
| **执行开发** | 节点可脚本化、可版本化、可接现有 pipeline API |
| **场馆/客户预览** | 网页里直接看 3D 覆盖与 VR180 立体效果 |

### 1.3 非目标（本期不做）

- 不重建已归档的账号/配额/通知平台层（`archive/platform-layer`）
- 不在 CI/测试里调真实 API 或下载模型
- 不做多人实时协作编辑
- 不做完整时间线 NLE（剪辑以镜头节点序为主，非 Premiere 级时间线）

---

## 2. 调研摘要（LibLibTV / ComfyUI / 同类产品）

### 2.1 调研方法与局限

- **ComfyUI 官方文档**已完整抓取：工作流=节点图、链接传数据、属性控参、JSON 存盘、队列/历史、侧栏（资产/节点库/模型/工作流/模板）、快捷键体系、App Mode、API 节点。
- **liblib.art** 为 SPA，静态抓取仅得到标题「LiblibAI-哩布哩布AI - 国内极具影响力的AI创作平台」，深度页面需登录/JS。以下 LibLib 模式来自其公开产品形态与国内同类（哩布哩布工作流社区）的通用认知，**实现前建议 owner 用账号再核对一次产品细节**。
- 「LibTV」在公开检索中无稳定独立站点；本 PRD 按 **LiblibAI 的视频/工作流创作形态** 理解，若 owner 指的是另一产品请指正。

### 2.2 可借鉴的界面模式

| 来源 | 模式 | 我们是否采用 |
|---|---|---|
| ComfyUI | 画布 + 节点 + 类型化端口 + 连线 | **是**，核心交互 |
| ComfyUI | 队列 / 历史 / 预览缩略图 | **是** |
| ComfyUI | 子图 / 模板工作流 | **是**（MVP 后） |
| ComfyUI | App Mode（复杂图收成简单面板） | **是**（给 owner 的「导演模式」） |
| ComfyUI | Partner/API 节点（云端付费模型） | **是**（MiniMax / Seedance / litellm） |
| LiblibAI | 模型/工作流社区分享、在线跑图、积分 | **部分**：本地工程文件 + 可导出 JSON；社区分享 P2 |
| LiblibAI | 中文优先 UI、参数面板 + 一键同款 | **是**（中文 UI；「一键套模板」P1） |
| 通用节点编辑器 | LiteGraph / Vue Flow / React Flow / Rete | 前端选型见 §7 |

### 2.3 与纯 ComfyUI 的差异（我们不能只是套壳 ComfyUI）

1. **输出目标是沉浸式成片**，不是单张图：节点图必须原生包含 VR180 / 球幕投影与元数据注入。
2. **脚本是一等公民**：分镜脚本、镜头描述、AI 润色，而不是只有 CLIP encode 文本框。
3. **3D 预览是验收门**：参数改完立刻看穹面覆盖 / 立体视差，而不是渲完再扔文件。
4. **复用本仓已验证的 pipeline**（`fulldome_mapper` / `stereo_renderer` / `check_source_quality` / usage ledger），不重写转换数学。

---

## 3. 产品愿景与成功指标

### 3.1 愿景

> 打开一个网页工程：左边写/润色分镜，中间画布连「首帧图 → 视频 → 超分 → 投影」，右边实时 3D 预览；
> 点「导出」同时得到 VR180 SBS 与 4096² domemaster。免费模型调构图，付费模型出成片。

### 3.2 成功指标（MVP）

| 指标 | 目标 |
|---|---|
| 从打开工程到出一版 5s 预览（mock/本地） | ≤ 15 分钟上手 |
| 改 prompt 后重跑受影响节点 | 只重算脏节点，不整图重跑 |
| 球幕覆盖问题发现时机 | **生成后、正式渲染前** |
| 供应商切换 | 画布节点属性里改 provider，无需改代码 |
| 密钥 | 不进 git；本地 settings / 环境变量 |

---

## 4. 用户旅程（主路径）

```
[新建工程]
   → 选模板「无人机前进 1:1 → VR180+球幕」
   → 脚本节点：场景描述 / 时长 / 运动
   → （可选）LLM 润色节点：litellm 改写为 VR180 友好 prompt
   → 首帧图节点：本地 litellm / SenseNova 或付费图模型
   → 视频节点：mock | local-litellm | MiniMax | Seedance
   → 质检节点：check_source_quality（方图/前进/锚点…）
   → 超分节点（可选）：SeedVR2
   → 分叉：
        ├─ VR180 节点链：深度 → 立体 → 等距 → sv3d/st3d
        └─ 球幕节点链：fulldome_mapper（4096² domemaster）
   → 3D 预览面板：穹面覆盖读数 / VR180 双目预览
   → 导出：mp4 + 工程 JSON + 账单摘要（usage ledger）
```

### 关键交互原则

1. **改一处，只重算下游**（图执行引擎按依赖脏标记）。
2. **花钱前必预览**：付费生成节点默认先出 draft 参数，正式提交二次确认。
3. **几何先于像素**：任何视频节点输出可先挂「覆盖/质检」预览，再进贵的超分/转换。
4. **失败可定位**：节点红框 + 错误原文（如 Seedance `ModelNotOpen`）。

---

## 5. 信息架构与界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│ 顶栏: 工程名 · 保存 · 撤销/重做 · 队列 · 预算/账单 · 设置(密钥)      │
├──────────┬─────────────────────────────────────┬─────────────────┤
│ 左侧栏    │           节点画布 (Canvas)           │  右侧检查器      │
│          │                                     │                 │
│ · 节点库  │   [脚本]──[LLM润色]──[首帧图]         │  选中节点属性     │
│ · 资产库  │        │              │             │  · provider     │
│ · 工作流  │        └──────────────┼──[视频生成]  │  · 模型/尺寸     │
│ · 模板    │                       │      │      │  · 费用预估      │
│          │                 [质检]─┴──[超分]     │                 │
│          │                      │         │    │  运行状态/日志    │
│          │            ┌─────────┴──┐  ┌───┴──┐ │                 │
│          │            │ VR180 链   │  │球幕链 │ │  质检摘要        │
│          │            └─────┬──────┘  └───┬──┘ │                 │
│          │                  └──────┬──────┘    │                 │
├──────────┴─────────────────────────┼───────────┴─────────────────┤
│ 底部抽屉: 队列 / 历史 / 资产预览 / 日志                            │
│ 右下浮动或右栏切换: 3D 预览 (WebGL)                                │
└──────────────────────────────────────────────────────────────────┘
```

### 5.1 左侧栏

| 面板 | 内容 |
|---|---|
| 节点库 | 按分类：脚本 / 生成 / 转换 / 质检 / 导出 / 工具；支持搜索 |
| 资产库 | 工程内图片/视频/音频；缩略图；拖入画布成「资产节点」 |
| 工作流 | 本机已存工程；打开/复制/重命名 |
| 模板 | 预设图：「1:1 前进源 → 双路导出」「球幕单帧验证」「纯脚本润色」 |

### 5.2 画布

- 平移（空格拖）、缩放、框选、对齐参考线
- 节点：标题栏、输入端口（左）、输出端口（右）、主体参数摘要、状态点（空闲/排队/运行/成功/失败）
- 连线：类型着色；不兼容端口禁止连接并提示类型
- 分组（Group）：把 VR180 链 / 球幕链框成组，可整组启停
- 快捷键对齐 ComfyUI 习惯：`Ctrl+Enter` 排队、`Ctrl+S` 保存、`Ctrl+Z/Y` 撤销重做、`Ctrl+G` 分组、`M` 静音节点

### 5.3 右侧检查器

- 选中节点时显示完整可编辑属性
- 付费节点显示：单价来源、本次预估费用、历史该节点花费
- 「应用并仅重跑此节点及下游」

### 5.4 3D 预览（与 dome-preview 对齐）

| 模式 | 内容 | 复用 |
|---|---|---|
| 球幕 | 半球内壁贴图 + 天顶角/半径读数 + 外圈覆盖率 | PR #376 `web/dome-preview` 的测量逻辑 |
| VR180 | 左右眼分屏 / 红青叠加；可调汇聚 | 新建，复用 equirect 采样思路 |
| 场馆参数 | 直径/倾角/投影仪布局示意（12m、5+3） | VENUE_ANALYSIS 结论，参数可配 |

**验收故事（沿用已发生事故）**：加载「内容只到 r≈0.6」的母版，读数必须明确显示外圈覆盖率 ≈8%。

---

## 6. 节点体系

### 6.1 端口类型（MVP）

| 类型 | 颜色建议 | 说明 |
|---|---|---|
| `text` | 灰 | 脚本/prompt 字符串 |
| `image` | 青 | 位图路径或内存引用 |
| `video` | 紫 | 视频路径/任务引用 |
| `json` | 黄 | 分镜/manifest/质检报告 |
| `number` | 蓝 | 时长、hfov、视差等 |
| `any` | 白 | 仅工具节点，慎用 |

### 6.2 节点目录（MVP 用 ★ 标出）

#### A. 脚本与文本

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| ★ 分镜脚本 | — | `text` + `json` | 场景、时长、运动、否定词；结构化字段可被下游读取 |
| ★ LLM 润色 | `text` + 指令 | `text` | 调 litellm（可配 base_url / model / api_key） |
| Prompt 模板 | 占位字段 | `text` | 封装 `pipeline/prompt_library.py` 模板 |
| 文本拼接 | 多 `text` | `text` | 模板套模板 |

#### B. 图像生成

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| ★ 文生图 | `text` + provider 配置 | `image` | provider: litellm-local / sensenova-u1 / 外部 |
| ★ 图生图/参考图 | `image` + `text` | `image` | 首帧锁定、改风格 |
| 资产加载 | — | `image`/`video` | 从工程资产库或本地路径 |

#### C. 视频生成

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| ★ 文生视频 | `text` | `video` | 供应商见 §8 |
| ★ 图生视频 | `image` + `text` | `video` | Seedance/MiniMax I2V 主路径 |
| ★ Mock 视频 | 参数 | `video` | 无 key 联通画布；对接 `MockProvider` |
| 任务轮询 | 任务 id | `video` | 异步 API 的提交/轮询拆分（可选进阶） |

#### D. 本仓转换与质检（核心差异化）

| 节点 | 输入 | 输出 | 调用 |
|---|---|---|---|
| ★ 源片质检 | `video`/`image` | `json` + 通过布尔 | `scripts/check_source_quality.py` |
| ★ SeedVR2 超分 | `video` | `video` | `pipeline/video_upscaler.py`（本地 CUDA） |
| ★ VR180 转换 | `video` | `video` | 深度→立体→等距→元数据（streaming/CLI 封装） |
| ★ 球幕转换 | `video`/`image` | `video`/`image` | `fulldome_mapper`（含 dome-* 参数） |
| 覆盖度测量 | `image`/`video` 帧 | `json` | 与 dome-preview 同源算法 |
| 导出成片 | `video` + 选项 | 文件 | 复制/封装到 `video/` 或用户指定目录 + 写 manifest |

#### E. 工具

| 节点 | 说明 |
|---|---|
| 预览 | 挂任意 image/video/text 端口，画布内联预览 |
| 笔记 | 画布便签 |
| 子图引用 | P1：把常用链收成可复用子图 |

### 6.3 图执行模型

- 有向无环图；节点一次「排队」= 从 Output/Export 节点回溯的闭包子图
- **缓存键** = 节点类型 + 归一化参数 + 上游输出内容哈希；命中则跳过
- 长任务（视频生成、超分）进入**队列**，支持取消；短任务同步
- 执行状态经 WebSocket/轮询推给画布

---

## 7. 技术架构（建议）

### 7.1 总览

```
浏览器 (React + 节点画布库)
    │  HTTP / WebSocket
    ▼
本地 FastAPI 网关 (本仓旁路 web/ 或 studio/)
    │
    ├─ 图执行引擎（拓扑排序 + 缓存 + 队列）
    ├─ Provider 适配层 ──► integrations/* + 新 litellm/sensenova/minimax
    ├─ Pipeline 封装   ──► pipeline/* + scripts/*
    ├─ 工程存储        ──► .studio/projects/*.json + 资产引用
    └─ 预计算服务      ──► 抽帧、覆盖度测量、缩略图
```

### 7.2 前端选型（待拍板）

| 选项 | 优点 | 缺点 |
|---|---|---|
| **A. React + @xyflow/react**（推荐） | 生态熟、自定义节点容易、与 WebGL 预览同栈 | 需自建执行状态同步 |
| B. Vue3 + Vue Flow | 与 ComfyUI 前端技术气质接近 | 团队若无 Vue 成本高 |
| C. 魔改 ComfyUI 前端 | 节点交互现成 | 与本仓 pipeline 深耦合困难；许可/耦合风险 |
| D. 纯自研 Canvas | 完全可控 | 成本过高，MVP 不划算 |

**默认建议 A**；3D 预览沿用 dome-preview 的 **无 CDN 纯 WebGL** 思路（离线可开）。

### 7.3 与现有代码的边界

| 现有 | Studio 中的角色 |
|---|---|
| `integrations/factory.py` | 供应商工厂；扩展 litellm/minimax/sensenova |
| `integrations/usage_ledger.py` | 每节点费用记账，对接右栏账单 |
| `integrations/minimax.py`（在 `feat/issue-353-minimax`，已 revert） | **可恢复**为节点后端，需重开 PR |
| `pipeline/prompt_library.py` | Prompt 模板节点数据源 |
| `scripts/check_source_quality.py` | 质检节点 |
| `scripts/run_pipeline.py` | 被包装为「VR180 转换」节点，不替代 CLI |
| `web/dome-preview`（PR #376） | 覆盖度测量与 3D 半球预览内核 |

### 7.4 工程文件格式

```json
{
  "version": 1,
  "name": "drone-1x1-dual-export",
  "nodes": [{"id": "n1", "type": "script.storyboard", "pos": [0,0], "params": {}}],
  "edges": [{"from": ["n1","text"], "to": ["n2","prompt"]}],
  "assets": [{"id": "a1", "path": "video/seed_1x1_drone.png"}],
  "settings": {"default_video_provider": "mock", "export": {"vr180": true, "dome": true}}
}
```

- 纯 JSON、可 diff、可进 git（密钥不入库）
- 媒体文件继续落在 `video/`（git 忽略），工程只存引用

### 7.5 密钥与配置

| 配置 | 位置 | 说明 |
|---|---|---|
| LiteLLM base_url / model / api_key | `studio/settings.json` 或环境变量 | 本地/自建网关 |
| SenseNova U1 | 同上 | **免费档验证**：需 owner 提供开通状态与模型 ID |
| MiniMax / Seedance API key | 同上 | 付费；UI 显示预估费用 |
| 不写入 | git 工程 JSON | 强制 |

---

## 8. 生成供应商策略

| 供应商 | 档位 | 状态 | 节点支持 |
|---|---|---|---|
| Mock | 免费 | 已有 `MockProvider` | ★ MVP |
| Local LiteLLM / 自建 | 免费/自担 | **待接** | ★ MVP（先打通 chat + 图，若支持视频则挂视频） |
| SenseNova U1 | 免费（用户称） | **待验证**：模型 ID、是否真免费、I2V 能力、限额 | ★ 验证卡 |
| Seedance | 付费 ~50 元/10s 4k | 已有集成 | ★ MVP |
| MiniMax H3 2K | 付费 ~9.5 元/10s | 代码在历史分支，**已 revert，待恢复** | P1 |
| Kling / Veo | 付费 | 已有集成 | 可选 |

**成本护栏（产品级）**：

1. 付费节点默认 draft 分辨率/时长；
2. 提交前 UI 显示预估费用（结合 ledger 历史）；
3. 单工程/单日软上限提醒（不强制扣款，只告警）。

### 8.1 SenseNova U1 验证卡（前置）

- [ ] 确认 LiteLLM 是否官方支持该 provider，或需自定义 `custom_llm_provider`
- [ ] 拿到模型列表；确认图像/视频/仅文本
- [ ] 跑通：同一 prompt 出图 → 作为 I2V 首帧 → 若支持视频则出 5s
- [ ] 记录限额、时延、分辨率上限；写入本 PRD 附录

#### 附录 · SenseNova / LiteLLM 接入状态（2026-09-17）

| 项 | 状态 |
|---|---|
| Studio 节点 `text.llm_polish` provider=`sensenova` | **已实现**：要求 `base_url` + `api_key` + `model`，走 OpenAI 兼容 `/chat/completions` |
| Studio 节点 `text.llm_polish` provider=`litellm` | **已实现**：同上，配置 `STUDIO_LITELLM_*` |
| 免费档是否覆盖 U1 文生图/图生视频 | **未查实** — 需 owner 提供控制台开通状态与模型 ID |
| CI | 仅 `provider=mock`（本地 `prompt_builder.wrap_prompt` 确定性润色），不调外网 |

环境变量（密钥不进 git / 不进工程 JSON）：

```
STUDIO_LITELLM_BASE_URL=...
STUDIO_LITELLM_API_KEY=...
STUDIO_LITELLM_MODEL=...
STUDIO_SENSENOVA_BASE_URL=...
STUDIO_SENSENOVA_API_KEY=...
STUDIO_SENSENOVA_MODEL=...
ARK_API_KEY=...   # Seedance
```

---

## 9. 导出规格

| 输出 | 规格 | 备注 |
|---|---|---|
| VR180 | SBS，方图/眼（默认对齐现有 2880² 档），`sv3d`/`st3d` | 复用管线 QA（`vr180_qa`） |
| 球幕 | **单张 4096² 圆形 domemaster**，不切分、不 warp | 场馆 HDX8 自标定（DECISION_DOME） |
| 工程 | `*.studio.json` + 资产清单 | 可再次打开继续改 |
| 账单 | 本工程各节点费用 CSV/JSON | usage_ledger |

导出节点可并行出双路；允许只导出一路。

---

## 10. 里程碑

> **2026-09-17 实现进度**（分支 `feat/node-studio-m0` / PR #380）
> M0 ✅ · M1 节点层 ✅（litellm/sensenova 需真实 key 联调）· M2 节点层 ✅（3D 半球 WebGL 预览仍依赖 #376 合入后嵌入）
> M3/M4 未开始。

### M0 — 骨架（1 个可运行故事）

- 本地 FastAPI + 静态前端；画布可加节点、连线、保存/加载 JSON
- 节点：脚本、Mock 视频、预览、导出（写文件）
- **不做**真实供应商

### M1 — 生成接线

- LiteLLM 聊天/润色节点
- SenseNova U1 验证结果落地（通过则挂文生图/图生视频；不通过则文档记录并降级）
- Seedance 文生/图生视频节点 + 费用预估
- 源片质检节点

### M2 — 沉浸式转换与 3D 预览

- 球幕转换节点 + 覆盖度测量 + 半球预览（吃进 dome-preview）
- VR180 转换节点（可先调 CLI 子进程，list-form，禁 shell=True）
- 双路导出

### M3 — 体验与成本

- MiniMax 恢复接入
- 脏节点重算与缓存
- 模板库 +「导演模式」简化面板
- 账单面板与软上限

### M4 — 增强（按需）

- 子图、工程分享、批量镜头、Blender 标准靶片节点（接 PLAN_DOME_AND_BLENDER）

---

## 11. 验收清单（MVP = M0+M1+M2）

- [ ] 无任何 API key 时，可用 Mock 节点走通「脚本→预览→导出占位」
- [ ] 配置 LiteLLM 后，润色节点改写分镜并在画布看到 diff
- [ ] SenseNova U1 有明确「支持/不支持」结论与记录
- [ ] Seedance 节点提交前显示预估费用；成功后资产进库（真实 key 由 owner 提供，CI 用 mock）
- [ ] 质检节点能复现现有脚本的方图/前进等判定
- [ ] 球幕节点输出 4096² 圆图；3D 预览对坏母版读出「外圈覆盖率低」
- [ ] VR180 节点产出可播放 SBS（或 mock 几何样片）+ 元数据存在
- [ ] 工程 JSON 不含密钥；`ruff` + `pytest -m "not slow"` 绿
- [ ] 前端可离线打开（无强制 CDN）

---

## 12. 风险与开放问题（需 owner 拍板）

| # | 问题 | 建议默认 |
|---|---|---|
| Q1 | 「LibTV」是否就是 LiblibAI 视频工作流？有无账号可再体验 | 按 LiblibAI 模式设计；有链接请给 |
| Q2 | SenseNova U1 的 LiteLLM 接入方式与真实免费额度 | 先开验证卡，不阻塞 Mock/Seedance |
| Q3 | MiniMax 是否恢复合并（曾 #373 后 revert #374） | **已决定：不恢复**。Owner：只到 2K、不能到 4K，整条线删除，缩短流程（2026-09-22 从本 PR 剔除） |
| Q4 | Studio 代码放哪：本仓 `studio/` 还是独立仓 | **建议本仓 `studio/`**，复用 pipeline 测试 |
| Q5 | 前端是否允许引入构建链（Vite/React） | **建议允许**；产物可 dist 进仓或构建脚本 |
| Q6 | VR180 节点是子进程调 CLI 还是 in-process API | MVP 子进程（稳）；M3 in-process |
| Q7 | 是否要兼容直接跑 ComfyUI 工作流 JSON | 否（P2 再评估导入器） |
| Q8 | 球幕静帧已通、视频链路未试，Studio 是否优先把「图→球幕视频」打穿 | 是，作为 M2 故事之一 |

---

## 13. 与当前主线任务的关系

| 线 | 状态 | Studio 关系 |
|---|---|---|
| 球幕 CLI `--dome-*` | 已合并 #375 | Studio 球幕节点调用它 |
| dome-preview PR #376 | 开放，可合并 | 作为 3D 预览内核前置合并 |
| #368 Apache 链路验证 | 开放 | 影响 VR180 节点默认深度/立体后端选择，不阻塞 Studio 骨架 |
| #372 anchor 面积门槛 | 开放 | 质检节点参数可配置，规则以 issue 为准 |
| MiniMax #353 | 已 revert | M3 恢复，服务付费备选 |

**推进顺序建议**：先合并 #376 → 开 Studio M0 任务卡 → SenseNova 验证卡（并行）→ M1/M2。

---

## 14. 附录 A — 界面文案原则

- 中文优先；技术词保留英文（VR180、domemaster、Seedance）
- 费用相关必须数字+单位（元/条、秒）
- 错误信息带「下一步怎么办」（缺 key、余额、模型未开通）

## 15. 附录 B — 参考

- ComfyUI Workflow / Interface docs（已抓取 2026-09-17）
- `docs/SOLUTION_ARCHITECTURE.md` · `DECISION_DOME.md` · `DECISION_MINIMAX.md`
- `docs/prompt-design-guide.md` · `pipeline/prompt_library.py`
- PR #376 `web/dome-preview`
- 历史分支 `feat/issue-353-minimax`（含 `integrations/minimax.py`）

---

## 16. 确认请求（请逐项回复）

请对下列默认项 **同意 / 修改**：

1. 产品名与范围：本仓内 `studio/` 节点式网页工作室（MVP：脚本+生成+双路导出+3D 预览）
2. 前端：React + xyflow；3D 用无 CDN WebGL
3. 供应商顺序：Mock → LiteLLM/润色 → SenseNova 验证 → Seedance → MiniMax(M3)
4. 代码位置：本仓，不新建独立仓
5. 先合并 dome-preview（#376）再开 M0 实现卡
6. 其它你点名但 PRD 未覆盖的需求（若有请写）

确认后我按 M0 拆 GitHub 任务卡并开始实现。
