# 下一轮持续开发 · 接班提示词套装（2026-09-12）

> 主干 `3d6de9e` · 本文件是给"另起对话"用的：每段都是可直接粘贴的提示词。
> 你同步要做的事在最后 §F。

---

## A. 接班巡检 + 串行合并（第 1 条，先粘贴这条）

```
你是本仓 Lead（Claude），接管 vr180-ai-pipeline 的持续开发。
工作目录 D:\Github\vr180-ai-pipeline，主干 main@3d6de9e。

先只读巡检，不写代码，报告以下事实（现场查，不读台账）：
1. .venv/Scripts/python.exe scripts/audit_agent_dispatch.py 的完整输出
2. gh pr list --state open + 每个 open PR 的 gh pr checks（351、352）
3. gh issue list --state open（应只有 #348、#322）
4. git worktree list + git status --short
5. video/ 下 quest_4k_fe180.mp4、quest_4k_crystal_fe180.mp4、quest_4k_anchor_fe180.mp4 是否存在，
   每个 ffprobe 确认 5760x2880 hevc 24fps + sv3d/st3d（用 scripts/vr180_qa.py 或 ffprobe）

已知结论（你复核，不要直接信）：
- PR #351（#348 anchor 碎片化）与 PR #352（#322 鱼眼假阴性）都改
  scripts/check_source_quality.py + tests/test_check_source_quality.py，
  必须串行合并（先 #351 后 #352），后者合前先 rebase 解决冲突。
- 两者 CI 当前 CLEAN，但以你现场查的 gh pr checks 为准。

只报告，不合并，等我说"合"再动手。
纪律：docs/DEV_PROCESS.md（Lead 不直接写实现，超几行就写卡派单；
边界锁：只在 D:\Github\vr180-ai-pipeline 内工作）；
docs/AGENT_DISPATCH.md（不维护台账，状态以 git/gh 为准；
Issue 留评论即心跳；分支先 push 再宣布完成；收尾清 worktree）。
测试门：前台跑 pytest -m "not slow" 全绿 + ruff 全绿才算数（docs/foreground-pytest 血训：
不要后台跑测试）。
```

## B. 合并执行（巡检确认后，逐条下发，一次一条）

```
合 PR #351（先合这个）：
1. gh pr checkout 351 切到分支，读 diff 确认只动
   scripts/check_source_quality.py + tests/test_check_source_quality.py，
   ANCHOR_MIN_AREA 保持 0.008 未动。
2. 前台跑：ruff check + pytest -m "not slow"（全绿才继续）。
3. squash 合并进 main，关 #348，在 Issue 回写 commit + CI URL + 专项结果。
4. 合完报告新 main hash。

合完 #351 且我确认后，再下发 #352：
---
合 PR #352（必须在 #351 之后）：
1. 先把 fix/issue-322-fisheye-forward rebase 到新 main，解决
   check_source_quality.py 的冲突（保留 #351 的合并逻辑 + #322 的成像圆检测，
   两套测试都要保留）。
2. 变异检验必做：撤掉合并逻辑→#348 第一条断言变红；
   撤掉成像圆检测→#322 圆形鱼眼合成用例变红。贴失败输出。
3. 前台全量 pytest -m "not slow" + ruff 全绿，gh pr checks 全绿，才 squash 合并，
   关 #322，回写 Issue。
```

## C. 剪枝 + 过期文件清理（合并完成后下发）

```
清理轮（只删已确认无用的，先列清单让我点头，逐批删）：
1. 本地已合入 main 的分支：git branch --merged main 逐条核对，
   worktree-agent-* 尸体用 git branch -d；远端已合的头
  （fix/issue-343-anchor-texture、fix/issue-345-anchor-floor、
   fix/issue-320-dry-run-no-io、fix/issue-319-fake-timeout-test、
   feat/issue-346-dome-quality）用 git push origin --delete。
   不要动 archive/platform-layer，不要动未合并的 WIP（先列出来问我）。
2. 空壳：.tmp_concat/out.mp4（12 字节）、x_vr180/、x_vr180_temp/、o/
   （空目录，先 ls 确认空再删）。
3. 根文档：RUNNING_TASKS.md、AGENTS_RUNNING.md 已过期（#346 写"待确认"实际已合；
   调研写"跑"实际已交付），合完后重写一份 20 行内的现状（main hash + open PR/Issue + 测试单名），
   或删掉只留 QUEST_TEST_4K_FUSED.md + SOURCE_RULES.md。
4. 旧测试单 QUEST_TEST_1X1.md / QUEST_TEST_1X1_R2.md / QUEST_TEST_FEATHER.md：
   本轮 4K 验收通过前不删；通过后按 docs/TEST_ASSET_INDEX.md 规则整组删
   （测试单 + 样片 + JSON/sidecar 一起，不只删视频）。
5. prompt 残留：GEN_PROMPT_1x1.txt / V2 标注作废，当前生产只留 V4（无人机）+ V3（水晶）。
每批删完跑 git status + 全量 pytest -m "not slow" 快速确认（或至少 --collect-only 计数不减）。
```

## D. 测试反馈驱动（等 Owner 回填 QUEST_TEST_4K_FUSED.md 后下发）

> ⚠️ 2026-09-12 owner 反馈已吸纳：旧三条不再测（F-drone 实为旧鸟、F-crystal 透明出局、
> Q3/Q4/Q5 通过）。旧对照表保留备查，新开卡只看下面 G-7。

```
Owner 已回填 QUEST_TEST_4K_FUSED.md 的 Q1–Q5（读文件原文，不要猜）。
按这个对照表开下一轮卡（一次只开一类）：
- 主体崩坏（Q2 有崩）→ 回生成：按 SOURCE_RULES 二"刚体优先"换主体
  （水晶/无人机/光球三选一），prompt 见 GEN_PROMPT_1x1_V4.txt / V3.txt，
  不碰管线。交付：新 4k 源 + 新 quest 成片 + 体检输出。
- 远景白区/跳变（Q3 有问题）→ 回 prompt：SOURCE_RULES 三，
  远景必须写具体景物，禁"明亮开口/纯白天空"。
- 河水静止（Q4 有问题）→ 回 prompt：SOURCE_RULES 四，逐项写动效。
- 边缘可见/渐变环（Q5 有问题）→ 开管线卡：调 feather 曲线，
  只许动 pipeline/equirectangular_mapper.py 相关 + 测试，一卡一 PR。
- 都好无短板 → 开 segment_concat CLI 卡：pipeline/segment_concat.py
  已实现但无 CLI，把 5s 片拼 15–25s（含 crossfade），补 CLI + 测试 + 文档。
每张卡按 docs/DEV_PROCESS.md 模板写 GitHub Issue（含"只许改哪些文件" + 变异检验 + Do NOT），
派单前让我过目。
```

### D-2（新增 2026-09-12）：G-7 真无人机成片转换（零额度，P0）

```
开一张 GitHub Issue（标题 G-7: V4 无人机源转 quest 成片 + 体检 + 推头显）：
- 背景：真无人机源 video/gen_1x1_4k_drone.mp4（V4 prompt，刚体+哑光不透明，
  9/9 00:27，40MB）尚无 quest 成片；旧 quest_4k_anchor_fe180.mp4 实为 anchor（鸟）源，
  与本卡无关。Q3/Q4/Q5 已通过，不重测。
- 步骤：
  1. 体检：python scripts/check_source_quality.py video/gen_1x1_4k_drone.mp4（贴输出）。
  2. 转码：与 quest_4k_anchor_fe180.json 同参数
     （--input-projection fisheye --fisheye-fov 180，5760×2880 hevc 24fps 10s，
     sv3d+st3d，QA 6/6），输出 video/quest_4k_drone_fe180.mp4 + 同名 .json。
  3. adb push 到 Quest 3（2G0YC1ZF7X006Z）/sdcard/Movies/，md5 对上。
  4. 出新迷你测试单（只 2 问）：Q1 机身是否笔直刚性 / Q2 旋翼是否有异常。
- 约束：不调 prompt、不重生成（零额度）；只许读调用管线，不改管线代码；
  文件只许新增 quest_4k_drone_fe180.mp4/.json，不许覆盖旧三条。
- 派单前让我过目。
```

## E. Backlog（Q1–Q5 都好之后，按序开卡，不一次全开）

> ⚠️ 2026-09-12 更新：Q3/Q4/Q5 已通过；主体线收敛 V4 无人机（G-7 先行）。
> D-2 仍被交付规范卡住；E-4 已开卡 #353 待派单。

```
1. G-7 真无人机成片转换（P0，零额度，先做，见 §D-2）
2. 平移/俯仰测量扩展（横滚已证 1–2°，不是眩晕主因，见 ACCEPTANCE_2026-09-09.md）
3. --stabilize（等 2 定位成因再做，否则盲修）
3. D-2 球幕 CLI（前置：Owner 的场馆三问答复，直径/倾角、warp归属、是否立体）
4. MiniMax 官方 API 验证（2K 约 9.5 元/10s vs Seedance 50 元，见 ACCEPTANCE §③-4）
5. 本地部署验证（暂缓，见 ACCEPTANCE 建议：只在"不按条计费"有价值时做）
```

---

## F. 你（Owner）同步完成的措施清单

- [ ] **测**：~~按 QUEST_TEST_4K_FUSED.md §2 回填 Q1–Q5（10 分钟）。~~
  ✅ 旧三条 2026-09-12 已结（Q3/Q4/Q5 通过，Q1/Q2 等新片）。
  新片（G-7 `quest_4k_drone_fe180.mp4`）出来后只答 2 问：机身是否笔直刚性 / 旋翼是否异常。
- [ ] **卡合并**：对新对话先发 §A，确认巡检数字与本文件一致，再说"合 #351"，
  合完确认 hash 再说"合 #352"。不同时下发两合。
- [ ] **预算护栏**（OWNER_BRIEF §A，仍未设）：PowerShell 跑一次
  [Environment]::SetEnvironmentVariable('VR180_ARK_PRICE_PER_MTOKEN','27.75','User');
  [Environment]::SetEnvironmentVariable('VR180_BUDGET_CAP','100','User')
  硬上限另去方舟控制台费用中心设。
- [ ] **球幕三问**（D-2 前置，OWNER_BRIEF §C）：~~直径/倾角、warp 归属、是否立体。~~
  ✅ owner 2026-09-12 四问全拍（12m / HDX8 对方做 / 单目 / A 全圆母版半面填黑，
  见 DECISION_DOME.md）。**D-2 仍不开工：缺对方交付规范（帧率/编码/音频），拿到才开。**
- [ ] **MiniMax 决策**（ACCEPTANCE §③-4）：~~要不要花 9.5 元验证一次 2K。~~
  ✅ owner 2026-09-12 已拍 A 验证，E-4 卡 #353 `stage:ready` 待派单（见 DECISION_MINIMAX.md §6）。
  本地部署暂缓；强化"Gemini 定稿→只花最后一棒"。
- [ ] **合并窗口内**：不要在仓库根手动加文件/改代码，避免与 rebase 冲突；
  反馈写进 QUEST_TEST_4K_FUSED.md §3 或 WORKLOG 留言区。

---

## G. 整合派单 prompt（2026-09-12 晚 · owner 直接粘到新对话派单用）

> G-7（无人机成片）由本对话 Lead 亲自执行中，**不需要派单**；成片出来会自动推 Quest 3
> 并按 QUEST_TEST_DRONE_G7.md 请你测 3 问。下面是**其余全部工作**的派单包。

### G-1 · 接班巡检（第一条，先粘）

```
你是本仓 Lead（Claude），接管 vr180-ai-pipeline 持续开发。工作目录 D:\Github\vr180-ai-pipeline。
只读巡检，不写代码，现场查（不读台账）：
1. .venv/Scripts/python.exe scripts/audit_agent_dispatch.py 完整输出
2. gh issue list --state open（应只有 #354 G-7、#353 E-4、#348、#322；G-7 由另一会话执行中，跳过）
3. gh pr list --state open + gh pr checks（应只有 #351、#352，均应 CLEAN）
4. git worktree list + git status --short
报告巡检结果后停下等指令。纪律：docs/DEV_PROCESS.md（lead 不写实现代码；边界锁：只在本仓内）；
docs/AGENT_DISPATCH.md（Issue 评论即心跳，先 push 再宣布完成，收尾清 worktree）；
测试必须前台跑（pytest -m "not slow" 全绿 + ruff 干净才算数）。
```

### G-2 · 串行合并（巡检确认后逐条下发，一次一条）

```
合 PR #351（先合这个）：
1. gh pr checkout 351，读 diff 确认只动 scripts/check_source_quality.py +
   tests/test_check_source_quality.py，ANCHOR_MIN_AREA 保持 0.008 未动。
2. 前台 ruff check + pytest -m "not slow" 全绿。
3. squash 合并，关 #348，Issue 回写 commit + CI URL。报告新 main hash。
合完且 owner 确认后，再发下一段：
---
合 PR #352（必须在 #351 之后）：
1. fix/issue-322-fisheye-forward rebase 到新 main，解决 check_source_quality.py 冲突
   （#351 的合并逻辑 + #322 的成像圆检测都保留，两套测试都保留）。
2. 变异检验：撤合并逻辑→#348 断言变红；撤成像圆→#322 用例变红，贴输出。
3. 前台全量 pytest -m "not slow" + ruff 全绿 + gh pr checks 全绿，squash 合并，关 #322。
```

### G-3 · 剪枝清理（合并完成后下发）

```
清理轮（先列清单等 owner 点头，逐批删，每批删完 git status + pytest --collect-only 计数不减）：
1. 本地已合 main 的分支（git branch --merged main 逐条核对后 git branch -d，重点 worktree-agent-*）；
   远端已合头 git push origin --delete：fix/issue-343-anchor-texture、fix/issue-345-anchor-floor、
   fix/issue-320-dry-run-no-io、fix/issue-319-fake-timeout-test、feat/issue-346-dome-quality。
   不动 archive/platform-layer；未合并 WIP 先列出来问。
2. 空壳：.tmp_concat/out.mp4（12 字节）、x_vr180/、x_vr180_temp/、o/（先确认空）。
3. RUNNING_TASKS.md、AGENTS_RUNNING.md 已过期：重写为 ≤20 行现状或删除
   （只留 QUEST_TEST_DRONE_G7.md / QUEST_TEST_4K_FUSED.md / SOURCE_RULES.md / NEXT_DEV_PROMPTS.md）。
4. GEN_PROMPT_1x1.txt / V2 标注作废（生产只用 V4 无人机 + V3 水晶）。
5. 旧测试单 QUEST_TEST_1X1/R2/FEATHER 暂留，等 4K 线验收后按 docs/TEST_ASSET_INDEX.md 整组删。
```

### G-4 · E-4 MiniMax 验证（owner 已拍板 A，卡 #353 stage:ready）

```
执行 issue #353（读原文照做，要点）：同一首帧 video/seed_1x1_drone.png + 同一 prompt
GEN_PROMPT_1x1_V4.txt，调 MiniMax 2K API 出 10s 一条（约 9.5 元，实付以账单为准，截图回报）；
跑 scripts/check_source_quality.py；走管线（fisheye-fov 180）出 quest 对照片；与 Seedance 4k 版
同屏比对 px/度、远景、水动效、边缘。owner 盲测。
Do NOT：不搭本地环境、不下载 H3 权重、不碰管线代码。
```

### G-5 · D-2 球幕（被外部阻塞，拿到规范才开工）

```
前置：owner 已把 DECISION_DOME.md 的 7 条缺口清单发给场馆方（12m / HDX8 对方做 warp /
单目 / 全圆母版半面填黑已拍板）。等对方回交付规范（至少帧率/编码/音频三项）。
拿到后开 D-2：只做 fisheye 输出分支（v360 domemaster 4096²+黑底+音频 sidecar），
不做 warp/MPCDI。当前不动。
```

### G-6 · owner 本人待办（不在对话里，提醒用）

- [ ] 把 DECISION_DOME.md「供应商缺口清单」7 条发给场馆方（D-2 唯一阻塞）。
- [ ] G-7 成片推上来后，按 QUEST_TEST_DRONE_G7.md 答 3 问（机身/旋翼/溪水）。
- [ ] 预算护栏仍未设：[Environment]::SetEnvironmentVariable('VR180_ARK_PRICE_PER_MTOKEN','27.75','User');
      [Environment]::SetEnvironmentVariable('VR180_BUDGET_CAP','100','User')；硬上限去方舟控制台。
