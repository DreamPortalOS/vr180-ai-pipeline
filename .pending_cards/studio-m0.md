Implements M0 of docs/PRD-node-studio.md.

## Goal
Local web app: canvas with typed nodes, save/load project JSON, no real paid APIs.

## Scope
- `studio/` FastAPI gateway + static frontend
- Node types: Script / Mock Video / Preview / Export
- Project file: `*.studio.json` (no secrets)
- Wire types: text, video, json
- Queue status is synchronous for M0
- Tests: CPU-only, no network, no model download

## Acceptance
- [ ] Open frontend, add nodes, connect, save, reload
- [ ] Mock video path produces a playable placeholder file
- [ ] Export writes project summary under `video/` or user path
- [ ] `ruff check` + `ruff format --check` + `pytest tests/ -m "not slow" -q` green
- [ ] No keys in git

Refs: PRD §10 M0, §11
