# CLAUDE.md — vr180-ai-pipeline 开发行为规范

你是本仓的开发执行者。任务来源是 **lead 交给你的任务卡**（GitHub Issue 或直接指派）。
卡片即规格：**只做卡里写的事**，验收标准全部满足 + CI 绿 = 完成。

## 项目一句话

把 2D 视频（含 AI 生成素材）转换成沉浸式视频：**路线1 Fulldome**（单目鱼眼 domemaster）
和 **路线2 VR180**（双眼 SBS 等距投影 + `sv3d`/`st3d`）。架构见
`docs/SOLUTION_ARCHITECTURE.md`，模块导览见 `docs/DEV_GUIDE.md`。

## 边界锁（违反 = PR 直接拒）

- 只改本仓文件；**不要**碰 `video/`、`models/`、`.venv/`（本地素材/权重/环境，git 忽略）。
- **不要**在测试或 CI 里下载模型、跑真实推理、调真实 API —— 一律 mock/注入伪后端。
  CI 是 ubuntu + CPU-only + 无模型；代码必须在这种环境下可 import、可测试。
- 不重建已归档的平台层（`archive/platform-layer` 分支）；不新增与卡片无关的依赖。
- 任务卡之外的"顺手改进"不要做；发现问题写进 PR 描述，由 lead 另开卡。

## 本地环境事实（写代码时的硬约束）

- Windows 主机：RTX 4070 SUPER **12GB**（SeedVR2 已部署，CUDA-only）；ffmpeg 在 PATH。
- Mac M2 Max（96GB 统一内存）：跑 DepthCrafter / StereoCrafter（MPS）。
- 因此：所有重模型封装必须**后端可插拔**（真实后端 + mock 后端），显存敏感参数
  （分辨率/批大小）必须可配置且默认值适配 12GB。

## 质量门（一票否决）

```bash
pip install -r requirements.txt
pip install pre-commit && pre-commit install --hook-type pre-commit --hook-type pre-push
ruff check . && ruff format --check . && pytest tests/ -m "not slow" -q
```

- 每个新模块/新分支逻辑配套 pytest 测试（mock 外部依赖，subprocess 必须 list 形式，禁 `shell=True`）。
- **绝不** `git commit --no-verify` / `git push --no-verify`。钩子红了就修到绿。
- ruff 版本以 `.pre-commit-config.yaml` 的 rev 为准（CI 已钉同版本），不得降级。
- 大文件（>150 行）用精准编辑，不要整文件重写。

## 提交与 PR

- Conventional Commits（`feat(scope): …` / `fix: …` / `test: …` / `docs: …`）。
- 一张卡 = 一个 PR；PR 描述里写清对应 issue 编号（`Closes #N`）与验收核对清单。
- PR 由自动评审（CI 绿 + AI 评审 APPROVE）squash 合并；评审要求改的照改，不争论。

### 增量提交纪律（防会话中途被切断，血泪教训）

网关会间歇性 `Connection closed mid-response` 把会话拦腰切断。已有多张卡因此**一行都没落地**，
最终报 `GraphQL: No commits between main and agent/issue-N`，白跑一整轮。因此：

- **写完一个文件就立刻 commit 一次**，不要攒到最后统一提交。会话随时可能断，
  只有已 commit 的部分保得住；重试时能接着做，而不是从零开始。
- commit 前**自己**先跑到全绿，不要指望 pre-commit 钩子替你修：
  `ruff check --fix . && ruff format . && ruff check . && ruff format --check .`
- push 前自检 `git log --oneline origin/main..HEAD`：**输出为空就不要建 PR**，
  先回头查为什么 commit 没落地。
- 如果发现自己「已经写了很多代码但一次都没 commit」，**立刻停下来先提交**。
