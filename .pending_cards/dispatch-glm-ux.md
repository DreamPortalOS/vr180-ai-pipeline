You are executing Studio UX card for D:\Github\vr180-ai-pipeline (Windows).

Model: glm-5.2 via company LiteLLM (already working).

READ: CLAUDE.md, docs/OVERNIGHT_2026-09-17.md, studio/static/app.js, studio/server.py.

Branch: git switch -c agent/glm-studio-ux

Implement:
1) Dirty-rerun UI: after selecting a node, a button「仅重跑此节点下游」 that POSTs /api/run with dirty_from=<nodeId> and same project payload.
2) Node status colors legend under canvas status bar.
3) In production template UI hint when backend is offline.

Tests: studio pytest suite green; no paid APIs.
ruff check/format first. Commit with paths. Push agent/glm-studio-ux.

Reply: **Status** / **Summary** / **Files** / **Validation**
