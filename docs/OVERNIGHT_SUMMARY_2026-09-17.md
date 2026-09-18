# 昨晚工作总结 · 2026-09-17 夜 → 09-18

**分支**：`feat/node-studio-m0` · **PR**：[#380](https://github.com/DreamPortalOS/vr180-ai-pipeline/pull/380) OPEN
**自动化**：全量 `pytest -m "not slow"` **3178 passed**；Studio 子集 **55 passed**
**测试清单**：`docs/STUDIO_TEST_CHECKLIST.md`

---

## 一、交付了什么

### 1. Immersive Node Studio（网页节点创作台）

| 能力 | 说明 |
|---|---|
| 节点画布 | 类 ComfyUI：脚本 / 分镜 / 生成 / 拼合 / 配乐 / 球幕 / VR180 / 导出 |
| **生产流程模板** | 项目脚本 → 分镜列表 → 批量润色 → 分镜图批出 → **审核门** → 图→视频 → 镜头拼合 → 配乐+音轨 → 质检/球幕覆盖度/VR180 → 导出 |
| **3D 球幕预览** | WebGL 半球 + 覆盖度黄环/红区留白；坏母版≈60°/r=0.6 可视化 |
| 画幅/场馆面板 | 1:1 源、VR180 单眼/SBS、domemaster、16:9；12m 场馆、5+3 投影、4096² 单目等 |
| 工程库 | 存库 / 打开 / 删除（`/api/projects`，密钥不进工程 JSON） |
| 脏重跑 | API `dirty_from` / `only_downstream_of` + UI 按钮 |
| 分镜图廊 | 运行后展示联络表与逐镜缩略图（`/api/media`） |

### 2. 生成 / 管线节点

| 节点 | 状态 |
|---|---|
| LiteLLM 润色 | ✅ 网关实测 **glm-5.2 / auto / internlm-s2**；默认 glm-5.2；429 自动重试 |
| Seedance | ✅ mock 默认；付费需 `confirm_paid` + 费用估算 |
| MiniMax | ✅ provider 已恢复（曾 #373 后 revert）；Studio `provider=minimax` |
| SeedVR2 超分 | ✅ `mode=mock`（lanczos）/ `mode=seedvr2`（CUDA 主机） |
| 球幕转换 + 覆盖度 | ✅ 对齐 domemaster 语义与场馆交付约定 |
| VR180 | ✅ `backend=apache`（商用安全：depth-anything+default+无 Crafter）/ `full` / `auto` |
| 源片质检 | ✅ 包装 `check_source_quality` |

### 3. 文档 / 过程

- `docs/PRD-node-studio.md` · `docs/OVERNIGHT_2026-09-17.md` · `docs/OVERNIGHT_HANDOFF.md`
- `docs/APACHE_CHAIN_FEASIBILITY.md` · `docs/LITELLM_LOCAL_NOTES.md`
- `docs/STUDIO_TEST_CHECKLIST.md` · README Studio 使用说明

### 4. Issue 关闭

| Issue | 结论 |
|---|---|
| #381 MiniMax 恢复 | ✅ 主线已交付 |
| #382 脏重跑 | ✅ API + UI |
| #383 分镜图廊 | ✅ gallery + 缩略图 |
| #384 G-11 anchor 重标 | ✅ main 上已是 `ANCHOR_TEXTURE_WINDOW=31`，质检测试绿 |

---

## 二、未做成 / 受阻

| 项 | 原因 |
|---|---|
| Claude Code 夜班代理 | `deepseek-pro` 网关 400/无 fallback；`glm-5.2 --bg` **ECONNRESET**；CLI `--model glm-5.2` **unrecognized_model** |
| SenseNova U1 付费生成联调 | 网关上 `sensenova-u1-fast` **不是 chat 模型组**（404）；图/视频 endpoint 未接 |
| #372 G-12 anchor 面积门槛 | 未做调研提案 |
| #368 Apache 商业化深化 | 节点参数已有；商用 profile 全链路对比未跑 |
| 分支清理 | 会话禁删共享 ref；需主检出跑 `tools/cleanup_branches.ps1` |

---

## 三、网关结论（开发用）

| 模型 | chat |
|---|---|
| **glm-5.2** / **auto** / **internlm-s2** | ✅ 可用 |
| deepseek-pro / sensenova-lite 等 | ⚠️ 常 429 |
| sensenova-u1-fast 等 | ❌ 非 chat group |

Base：`http://49.235.157.202:9400`（密钥只进本机 settings，不进 git）

---

## 四、请你批注的单子

见 GitHub：

- **决策单** `decision: …`
- **测试单** `test: …`

反馈格式建议：在 issue 里直接勾选 / 写「同意 / 改为 …」。
