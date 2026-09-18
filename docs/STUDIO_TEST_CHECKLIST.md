# Studio 手工/联调测试清单（feat/node-studio-m0）

> 自动化基线：`pytest tests/ -m "not slow" -q` → **3178 passed**（分支最新）
> Studio 子集：**55 passed**（本清单对应模块）

## A. 环境（5 分钟）

| # | 步骤 | 期望 |
|---|---|---|
| A1 | 启动 `.venv\Scripts\python.exe -m studio.server` | 进程存活；**不要**用系统 `python` 起旧实例 |
| A2 | 打开 http://127.0.0.1:8787 | 深色 UI；侧栏「已连接」+ work_root 路径 |
| A3 | `GET /api/health` | 含 `projects_root` / `settings_path` |
| A4 | `GET /api/node-types` | 含 `script.shot_list`、`video.concat`、`video.seedvr2`、`convert.vr180`、`text.llm_polish` |
| A5 | 主 UI 标签 | 检查器 / 3D 球幕 / 画幅场馆 / 工程 |

## B. 生产流程模板（核心）

| # | 步骤 | 期望 |
|---|---|---|
| B1 | 点「生产流程」 | 画布出现：项目脚本→分镜列表→润色→分镜图批出→审核门→视频→拼合→配乐→音轨→球幕/VR180→导出 |
| B2 | 「▶ 运行」 | 全部节点变绿；状态图例有 ok |
| B3 | 运行结果 | **分镜图廊**：联络表缩略图 + 每镜缩略图 + 路径 |
| B4 | 产物 | work_root 下 `studio_export/n_export/final_master.mp4` 存在 |
| B5 | 二次运行 | 大量 `cache_hit=true`（未改参时） |
| B6 | 选中节点→「仅重跑此节点下游」 | 仅 dirty 链重算；无关节点 skipped/cache |

## C. 3D 与场馆

| # | 步骤 | 期望 |
|---|---|---|
| C1 | 标签「3D 球幕」 | WebGL 半球可见；黄线覆盖环；红区留白 |
| C2 | 「坏母版预设」 | 读数 ≈54° / r≈0.6，level=bad |
| C3 | 「合规预设」 | 读数 ≥85° 量级，level=ok |
| C4 | 加载自己的分镜图/母版 | 浏览器内测覆盖度（http 打开，非 file://） |
| C5 | 运行后若含 `qa.dome_coverage` | 3D 面板自动吃进 coverage 报告 |
| C6 | 「画幅/场馆」 | 1:1 源 / VR180 / domemaster / 16:9 + 12m、5+3、4096²、单目等参数 |

## D. 本地 LiteLLM / glm-5.2

| # | 步骤 | 期望 |
|---|---|---|
| D1 | 标签「工程」→「本地 LiteLLM / 供应商」 | 表单可见 |
| D2 | 填 Base=`http://49.235.157.202:9400`，Model=`glm-5.2`，Key=网关 token | 「保存设置」成功；key 显示为 mask |
| D3 | 画布加 `LLM 润色`，provider=litellm，model 可空 | 节点参数可读 |
| D4 | 连到分镜 prompt 并运行 | 润色输出带 immersive 约束（continuous take / wide FOV…） |
| D5 | 缺 key 时运行 litellm | 错误信息含 base_url/api_key 配置提示 |
| D6 | provider=mock | 无网也可润色（本地规则） |

> 注意：网关 chat 对 **deepseek-pro / sensenova-lite** 常 429；`sensenova-u1-fast` **不是 chat 模型组**。开发默认 **glm-5.2 / auto**。

## E. 生成节点（默认 mock，控费）

| # | 步骤 | 期望 |
|---|---|---|
| E1 | `video.seedance` provider=mock | 产出 mp4；cost=0 |
| E2 | provider=seedance 且 confirm_paid=false | 拒绝提交，提示费用估算 |
| E3 | provider=minimax 且无 confirm | 同样拦截；费用表含 minimax-2k≈9.5 元/10s |
| E4 | `video.seedvr2` mode=mock | lanczos 放大，文件变大 |
| E5 | mode=seedvr2 无 CUDA/权重 | 明确报错并提示 mock / setup_seedvr2 |

## F. 工程库

| # | 步骤 | 期望 |
|---|---|---|
| F1 | 「存库」 | 状态栏出现工程 id |
| F2 | 「工程」刷新 | 最近工程列表（name/nodes/mtime） |
| F3 | 打开某工程 | 画布恢复节点与连线 |
| F4 | 删除 | 列表移除；404 语义正确 |
| F5 | `GET /api/projects` | JSON 数组 |

## G. 转换与质检

| # | 步骤 | 期望 |
|---|---|---|
| G1 | `convert.dome` size=128 + 运行 | 输出 domemaster mp4 |
| G2 | `qa.dome_coverage` | report 含 coverage_deg / level |
| G3 | `convert.vr180` mode=mock | 文件透传 |
| G4 | mode=cli backend=apache | 调 run_pipeline；meta.backend=apache（真渲需时间/GPU） |
| G5 | `qa.source_quality` mode=mock/auto | report 结构完整 |

## H. 回归 / 交付

| # | 命令/动作 | 期望 |
|---|---|---|
| H1 | `ruff check studio` | clean |
| H2 | 全量 `pytest -m "not slow"` | **≥3178 passed** |
| H3 | `git log origin/main..HEAD` | 多条 feat(studio) 提交 |
| H4 | PR #380 | OPEN，可审可合 |
| H5 | PR #376 dome-preview | 与 Studio 3D 算法同源，建议先合 |
| H6 | 主检出 `tools/cleanup_branches.ps1` | 清僵尸分支（owner 执行） |

## I. 已知限制（测试时勿当 bug）

1. 分镜图/多数视频节点默认 **mock**；付费需 key + confirm_paid
2. `deepseek-pro` 网关 **429**；Claude `--bg` 曾 ECONNRESET
3. VR180 `mode=cli` 全链路慢；Studio 默认小尺寸/少帧
4. file:// 直开 HTML 时覆盖度可能 tainted → 用 http://127.0.0.1:8787
5. secrets 只在 work_root settings，工程 JSON/git 无 key

## J. 推荐验收顺序（约 30–40 分钟）

1. A1–A5 环境
2. B1–B4 生产流程一次跑通 + 图廊缩略图
3. D2–D4 glm-5.2 润色
4. C1–C2 3D 坏母版读数
5. B5–B6 缓存与脏重跑
6. F1–F3 存库/打开
7. E1 / E4 mock 生成与超分
8. H2 全量 pytest（可放最后）
