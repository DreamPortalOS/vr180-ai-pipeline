You are executing overnight Studio cards issues #382 (dirty rerun) and #383 (stills gallery) for D:\Github\vr180-ai-pipeline.

READ: CLAUDE.md, docs/OVERNIGHT_2026-09-17.md, docs/PRD-node-studio.md, studio/ package.

Branch: `git switch -c agent/overnight-studio` from current HEAD if needed, or create from origin/feat/node-studio-m0:
`git fetch origin && git switch -c agent/overnight-studio origin/feat/node-studio-m0`

Implement:
1) studio/graph.py — support only_downstream_of: when set, nodes not in {target}∪ancestors get status skipped without executing; cache still works. Update tests in tests/test_studio_graph.py.
2) Studio run panel UX: after /api/run, if results contain stills/sheet (n_stills outputs sheet + meta), display contact sheet path and shot list in run panel (studio/static/app.js + index.html if needed).
3) Optional small: server returns gallery block in run response when present.

Tests: studio suite + pytest tests/ -m "not slow" must go green in foreground before finish.
No real paid APIs. No model downloads. Mock only.

Commit paths: studio/..., tests/...
Push: git push -u origin agent/overnight-studio

Reply format:
**Status** / **Summary** / **Files touched** / **Validation** / **Findings**
