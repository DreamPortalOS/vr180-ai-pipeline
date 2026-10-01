# CLI_WORKERS — 有界 CLI 派单启动与判活（2026-10-02）

本文件说明本仓 worker（Cline / Claude Code CLI）的有界启动、判活和交接。事实源仍是 GitHub Issue/PR + git
（见 `docs/AGENT_DISPATCH.md`）；**不建 JSON 任务台账**，Issue 评论即现场记录。
范围仅限本仓：不读外部编排系统的配置、状态或日志目录。

## 1. 启动前（每次都要做）

- 用**独立 worktree**，不在主检出里改东西：
  `git worktree add .claude/worktrees/issue-<N>-<slug> -b codex/issue-<N> origin/main`
- 任务卡写进 **ignored 的 `output/`**（`.gitignore` 已含 `output/`），不提交。
- 使用任务专用变量（`$TaskFile` / `$WorkerTree` / `$WorkerOut`）；不挪用 HOME/CODEX_HOME，不改全局模型/认证配置。
- 代理与超时只在**当前进程**生效（`$env:`）；不写用户/系统级环境变量，不落盘，不打印 key。

```powershell
$TaskFile   = "output/cli-discipline-task.md"          # ignored；任务卡正文作为 prompt 传入
$WorkerTree = "D:/Github/vr180-ai-pipeline/.claude/worktrees/issue-450-cli-workers"
$WorkerOut  = "output/issue-450"                       # ignored
New-Item -ItemType Directory -Force -Path $WorkerOut | Out-Null

$env:HTTP_PROXY  = "http://127.0.0.1:7897"
$env:HTTPS_PROXY = "http://127.0.0.1:7897"
$env:NO_PROXY    = "localhost,127.0.0.1,::1"
# 先检查支持情况；不要先追加一个旧 Node 无法识别的选项。
if (-not ((node --help | Out-String).Contains('--use-env-proxy'))) { throw 'Node 不支持环境代理' }
# 追加，不覆盖已有 NODE_OPTIONS（Node 需该 flag 才会读代理；已含则不重复加）
if ($env:NODE_OPTIONS -notmatch '--use-env-proxy') {
  $env:NODE_OPTIONS = (@($env:NODE_OPTIONS, "--use-env-proxy") |
    Where-Object { $_ }) -join " "
}
cline --version                                          # 先现场核对版本，不要凭记忆写死
$CliTaskPrompt = Get-Content -Raw -Encoding UTF8 $TaskFile
cline --json --cwd $WorkerTree --timeout 180 --retries 1 $CliTaskPrompt 1> "$WorkerOut/stdout.jsonl" 2> "$WorkerOut/stderr.log"
$LASTEXITCODE
```

- 已装 **Cline 3.0.65** 上 **`--retries 0` 是非法值，最小为 1**；本机 2026-10-02 `cline --version` 为
  3.0.67，所以**每次现场核对版本**，不写死。`--retries` 是"连续错误次数上限"（默认 6），**不是**
  进程重启次数；`--timeout` 默认 0 = 不限时。**外面不要再套自动重启循环**，超时后按第 3 节人工判定。
- 不传 `-k/--key`、不打印 key；不重定向 HOME/CODEX_HOME、Cline 状态目录或全局认证配置。

## 2. 环境和连通性快照（2026-10-02）

- Node **v22.22.3**，`node --help` 含 `--use-env-proxy`；`output/` 被忽略，`.claude/` **未**被忽略
  （所以 worktree 目录会出现在未跟踪列表里，提交必须带路径，禁止 `git add .`）。
- 同一代理档下：两次**不带代理**的健康检查超时，第三次**带代理**在 **29984 ms** 内返回 OK。
- 后续 #450 文档任务完成了两文件交付，但设置 180s 后报告 durationMs=212919；CLI 超时不单独作为硬时限。
- 上述命令是前台调用示例。生产派单须由外部 supervisor 记录所创建 PID/启动时间/命令/worktree，并按墙钟限时；
  到期先确认该进程仍归本任务所有，再终止该进程及其自建子进程、保存日志。观测超时本身不是终止理由。
- 健康 OK 只证明连通性；文件交付仍需独立验收，费用元数据不等于账单核验。

## 3. 判活与交接（三态分清）

| 现象 | 判定 | 动作 |
|---|---|---|
| PID 在、worktree 时间戳更新、日志在长 | **活跃** | 等；到任务卡 deadline 才干预 |
| 本机观测超时，但远端 CI 仍 pending | **未定**（≠失败） | 查 Issue/PR/`gh pr checks`，不重派 |
| 进程已退出、无提交、日志停在失败原因 | **终态** | 记录 reason，进入交接/接管 |
| 分支 4h 无 commit/无 Issue 评论/无 PR 更新 | **stale** | 停额度，人工检查 |

- 任务卡必须写清：**文件范围、验收命令、CLI deadline、PID/worktree/结果的新鲜度判据、终态 reason**。
- 未到任务硬时限或未核对归属时，不终止活跃进程；不编辑活跃 worker 的 worktree。
- 接管前先确认**终态归属**：owner 已退出或明确停止、PID 属主是本任务、worktree/分支归属清楚。
- 超时、无效或不合格交付：保全日志/diff、exit code/终态 reason 后，lead 按 owner 授权接手修复。

## 4. 合并门

- **一 issue 一 PR**；提交**带路径** `git commit -- <本卡文件>`，绝不 `--no-verify`。
- 合并前：`git ls-remote origin <branch>` 核对远端 SHA = 本地 HEAD。
- **当前 head** 的 lint / 测试 / e2e 全绿 + lead **显式 APPROVE** 才 squash 合并。
- 开工、阻塞、完成、移交各在 Issue 留一条评论；不把 key、模型路径、本地素材路径写进 Issue。

## 5. Claude Code CLI（2026-10-02 独立复测）

- 早先的有界审计以 `error_max_turns` 退出且**没有交付**；这**不**能证明模型拒绝，也**不**能证明当前 CLI 健康。
- 现场 ANTHROPIC_MODEL=mimo-v2.5-pro 返回 API 400 模型名无效；单次加 `--model glm-5.2` 后健康 OK（6.75s）。
- 主控用 `--bare --strict-mcp-config --tools Read --allowedTools Read --max-turns 4 --model glm-5.2 --output-format json`
  配合 `--print` 仅读取本仓 CLAUDE.md：2 轮、exit=0、is_error=false、首标题匹配、无权限拒绝。
- 两次检查均由 Python subprocess 外部 supervisor 限 60s；只证明健康/读取能力，不是业务开发验收或额度证明。
- 保留全局模型/认证配置；SDK 的 unrecognized_model 警告在成功调用中也存在，应以 API 结果/is_error/产物判断。
