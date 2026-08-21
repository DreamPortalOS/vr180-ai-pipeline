"""Job manifest for cross-machine staged pipeline (issue #36, V-3).

A manifest is a JSON file that records, for one conversion job:

  - ``job_id``: unique identifier for the job.
  - ``source_hash``: SHA-256 of the source video (hex digest).
  - ``stages``: one entry per pipeline stage with ``name``, ``status``
    (``pending`` / ``done``), ``inputs`` / ``outputs`` path lists,
    ``params`` (key stage parameters), ``machine`` (label of the machine
    the stage ran on, e.g. ``win-cuda`` / ``mac-mps``) and ``hashes``
    (SHA-256 of every output artifact).

This enables the Windows-CUDA ↔ Mac-MPS relay workflow: run a subset of
stages on one machine, copy the intermediate files + manifest to the
other machine, and resume with ``--resume-from`` after hash validation.

No network transfer is implemented here — copying files is manual.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Canonical stage names used in manifests (subset selection + resume).
STAGE_NAMES = ("upscale", "depth", "stereo", "project", "encode")

# Machine labels by convention (free-form strings are allowed).
MACHINE_WIN_CUDA = "win-cuda"
MACHINE_MAC_MPS = "mac-mps"

STATUS_PENDING = "pending"
STATUS_DONE = "done"

_HASH_CHUNK = 1024 * 1024  # 1 MiB


class ManifestError(Exception):
    """Raised when a manifest is invalid or hash validation fails."""


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of a file (streamed, 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths: list[str | Path]) -> dict[str, str]:
    """Hash every existing regular file in *paths*.

    Directories are skipped (intermediate frame dirs hold many files and
    are validated per-stage by the caller instead).  Missing files are
    silently omitted — validation happens in :func:`validate_stage_outputs`.
    """
    result: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if p.is_file():
            result[str(p)] = sha256_file(p)
    return result


def new_manifest(
    job_id: str,
    source_path: str | Path,
    stage_names: list[str] | tuple[str, ...] = STAGE_NAMES,
    machine: str | None = None,
) -> dict:
    """Create a fresh manifest with all stages pending."""
    return {
        "version": 1,
        "job_id": job_id,
        "source": str(source_path),
        "source_hash": sha256_file(source_path),
        "stages": [
            {
                "name": name,
                "status": STATUS_PENDING,
                "machine": machine,
                "inputs": [],
                "outputs": [],
                "params": {},
                "hashes": {},
            }
            for name in stage_names
        ],
    }


def load_manifest(path: str | Path) -> dict:
    """Load and minimally validate a manifest JSON file."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"Manifest is not valid JSON: {path}: {e}") from e
    if not isinstance(data, dict) or "stages" not in data or "job_id" not in data:
        raise ManifestError(f"Manifest missing required keys (job_id/stages): {path}")
    if not isinstance(data["stages"], list):
        raise ManifestError(f"Manifest 'stages' must be a list: {path}")
    return data


def save_manifest(manifest: dict, path: str | Path) -> None:
    """Write manifest to disk (pretty-printed, UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("💾 Manifest saved → %s", path)


def get_stage(manifest: dict, name: str) -> dict | None:
    """Return the stage entry with the given name, or None."""
    for stage in manifest.get("stages", []):
        if stage.get("name") == name:
            return stage
    return None


def mark_stage_done(
    manifest: dict,
    name: str,
    *,
    machine: str | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    params: dict | None = None,
) -> dict:
    """Mark a stage done, hashing existing output files into ``hashes``.

    Creates the stage entry if absent.  Returns the updated stage entry.
    """
    stage = get_stage(manifest, name)
    if stage is None:
        stage = {"name": name}
        manifest.setdefault("stages", []).append(stage)
    stage["status"] = STATUS_DONE
    if machine is not None:
        stage["machine"] = machine
    stage["inputs"] = [str(p) for p in (inputs or [])]
    stage["outputs"] = [str(p) for p in (outputs or [])]
    stage["params"] = dict(params or {})
    stage["hashes"] = hash_paths(stage["outputs"])
    return stage


def validate_stage_outputs(manifest: dict, name: str) -> None:
    """Verify a done stage's recorded output hashes still match on disk.

    Raises :class:`ManifestError` with a clear message when a recorded
    output is missing or its hash no longer matches (artifact corrupted
    or replaced since the manifest was written).
    """
    stage = get_stage(manifest, name)
    if stage is None:
        raise ManifestError(f"Stage '{name}' not present in manifest")
    if stage.get("status") != STATUS_DONE:
        raise ManifestError(f"Stage '{name}' is not marked done (status={stage.get('status')!r})")
    for path_str, expected in (stage.get("hashes") or {}).items():
        p = Path(path_str)
        if not p.is_file():
            raise ManifestError(f"Stage '{name}': recorded output missing: {path_str}")
        actual = sha256_file(p)
        if actual != expected:
            raise ManifestError(
                f"Stage '{name}': hash mismatch for {path_str}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                "  → artifact changed since manifest was written; re-run that stage."
            )


def completed_stages(manifest: dict) -> list[str]:
    """Names of stages marked done, in manifest order."""
    return [s["name"] for s in manifest.get("stages", []) if s.get("status") == STATUS_DONE]


def validate_source(manifest: dict, source_path: str | Path | None = None) -> None:
    """Verify the source file hash matches the manifest (if recorded)."""
    expected = manifest.get("source_hash")
    if not expected:
        return
    src = Path(source_path or manifest.get("source") or "")
    if not src.is_file():
        raise ManifestError(f"Source file not found: {src}")
    actual = sha256_file(src)
    if actual != expected:
        raise ManifestError(
            f"Source hash mismatch for {src}\n  expected: {expected}\n  actual:   {actual}\n"
            "  → the input video differs from the one the manifest was created for."
        )
