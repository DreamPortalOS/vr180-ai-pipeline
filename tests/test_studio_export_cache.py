"""Export cache receipts must still describe materialized files (issue #448)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from studio.graph import GraphError, run_graph
from studio.models import EdgeSpec, NodeSpec, Project
from studio.nodes.base import StudioNode
from studio.nodes.registry import NODE_REGISTRY


class LocalSource(StudioNode):
    def run(self, *, params, inputs, work_dir, node_id):
        return {"video": params["path"]}


@pytest.fixture
def setup_export(tmp_path, monkeypatch):
    monkeypatch.setitem(NODE_REGISTRY, "test.cache_source", LocalSource)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original local bytes")
    project = Project(
        nodes=[
            NodeSpec(id="source", type="test.cache_source", params={"path": str(source)}),
            NodeSpec(id="export", type="export.bundle", params={"filename": "original.mp4"}),
        ],
        edges=[EdgeSpec(id="edge", from_node="source", from_port="video", to_node="export", to_port="video")],
    )
    cache: dict[str, dict[str, Any]] = {}

    def run(work_root=None):
        return run_graph(project, work_dir=str(work_root or tmp_path), cache=cache).results[project.nodes[1].id]

    return source, project, cache, run


def assert_receipt(result, source):
    manifest = result.outputs["manifest"]
    disk = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert disk == {key: value for key, value in manifest.items() if key != "manifest_path"}
    dest = Path(result.outputs["path"])
    assert dest.read_bytes() == source.read_bytes()
    assert disk["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_unchanged_export_keeps_cache_hit(setup_export):
    source, _, _, run = setup_export
    assert not run().cache_hit
    second = run()
    assert second.cache_hit
    assert_receipt(second, source)


def test_filename_a_b_a_restores_manifest(setup_export):
    source, project, _, run = setup_export
    run()
    project.nodes[1].params["filename"] = "renamed.mp4"
    run()
    project.nodes[1].params["filename"] = "original.mp4"
    restored = run()
    assert not restored.cache_hit
    assert Path(restored.outputs["path"]).name == "original.mp4"
    assert_receipt(restored, source)


def test_source_a_b_a_restores_overwritten_bytes(setup_export, tmp_path):
    source, project, _, run = setup_export
    run()
    other = tmp_path / "other.mp4"
    other.write_bytes(b"different source bytes")
    project.nodes[0].params["path"] = str(other)
    run()
    project.nodes[0].params["path"] = str(source)
    restored = run()
    assert not restored.cache_hit
    assert_receipt(restored, source)


def test_changed_source_at_same_path_invalidates(setup_export):
    source, _, _, run = setup_export
    run()
    source.write_bytes(b"modified local bytes")  # same length as the old source
    modified = run()
    assert not modified.cache_hit
    assert_receipt(modified, source)


@pytest.mark.parametrize("artifact", ["video", "manifest"])
@pytest.mark.parametrize("action", ["delete", "tamper"])
def test_missing_or_tampered_artifact_recomputes(setup_export, artifact, action):
    source, _, _, run = setup_export
    first = run()
    path = Path(first.outputs["path"] if artifact == "video" else first.outputs["manifest"]["manifest_path"])
    if action == "delete":
        path.unlink()
    else:
        path.write_bytes(b"corrupted artifact")
    repaired = run()
    assert not repaired.cache_hit
    assert_receipt(repaired, source)


@pytest.mark.parametrize("scope", ["work_root", "node_id"])
def test_shared_cache_must_not_return_another_export_location(setup_export, tmp_path, scope):
    source, project, _, run = setup_export
    old = run()
    if scope == "node_id":
        project.nodes[1].id = "other_export"
        project.edges[0].to_node = "other_export"
        result = run()
    else:
        result = run(tmp_path / "other_root")
    assert not result.cache_hit
    assert result.outputs["path"] != old.outputs["path"]
    assert_receipt(result, source)


def test_legacy_cache_without_hash_recomputes(setup_export):
    source, _, _, run = setup_export
    first = run()
    first.outputs["manifest"].pop("sha256")
    repaired = run()
    assert not repaired.cache_hit
    assert_receipt(repaired, source)


@pytest.mark.parametrize("failure", ["copy", "metadata", "hash", "json"])
def test_failed_reexport_preserves_old_bundle(setup_export, monkeypatch, failure):
    from studio.nodes import export

    source, _, cache, run = setup_export
    first = run()
    dest = Path(first.outputs["path"])
    receipt = Path(first.outputs["manifest"]["manifest_path"])
    old_video, old_receipt = dest.read_bytes(), receipt.read_bytes()
    old_cache = dict(cache)
    source.write_bytes(b"changed source to force rematerialization")
    if failure == "copy":
        original_copy = export.shutil.copy2

        def broken_copy(src, dst):
            if Path(src) != source:
                return original_copy(src, dst)
            Path(dst).write_bytes(b"partial")
            raise OSError("injected partial copy")

        monkeypatch.setattr(export.shutil, "copy2", broken_copy)
    elif failure == "hash":
        original_hash = export._sha256

        def broken_hash(path):
            if path.name == "video":
                raise OSError("injected staged hash failure")
            return original_hash(path)

        monkeypatch.setattr(export, "_sha256", broken_hash)
    elif failure == "json":
        original_json = export.json.dumps

        def broken_json(value, *args, **kwargs):
            if isinstance(value, dict) and "export_path" in value:
                raise ValueError("injected manifest serialization failure")
            return original_json(value, *args, **kwargs)

        monkeypatch.setattr(export.json, "dumps", broken_json)
    else:
        original_write = Path.write_text

        def broken_metadata(path, *args, **kwargs):
            if path.name == "manifest.json":
                path.write_bytes(b"partial metadata")
                raise OSError("injected metadata failure")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", broken_metadata)
    with pytest.raises(GraphError, match="injected"):
        run()
    assert dest.read_bytes() == old_video
    assert receipt.read_bytes() == old_receipt
    assert cache == old_cache
    assert set(p.name for p in dest.parent.iterdir()) == {dest.name, "manifest.json"}


@pytest.mark.parametrize("external_write", [False, True])
def test_final_manifest_failure_rolls_back_only_owned_video(setup_export, monkeypatch, external_write):
    from studio.nodes import export

    source, _, _, run = setup_export
    first = run()
    dest = Path(first.outputs["path"])
    receipt = Path(first.outputs["manifest"]["manifest_path"])
    old_video, old_receipt = dest.read_bytes(), receipt.read_bytes()
    source.write_bytes(b"new export payload")
    replace = export.os.replace

    def fail_final_metadata(src, dst):
        if Path(dst) == receipt:
            if external_write:
                dest.write_bytes(b"unknown actor bytes")
            raise OSError("injected final metadata replace")
        return replace(src, dst)

    monkeypatch.setattr(export.os, "replace", fail_final_metadata)
    with pytest.raises(GraphError, match="recovery retained" if external_write else "injected"):
        run()
    assert receipt.read_bytes() == old_receipt
    if external_write:
        assert dest.read_bytes() == b"unknown actor bytes"
        (recovery,) = dest.parent.glob(".export-*")
        assert (recovery / "previous-video").read_bytes() == old_video
        assert (recovery / "previous-manifest").read_bytes() == old_receipt
    else:
        assert dest.read_bytes() == old_video
        assert not list(dest.parent.glob(".export-*"))
    assert not (dest.parent / ".export.lock").exists()


def test_existing_lock_is_not_removed(setup_export):
    source, _, _, run = setup_export
    first = run()
    dest = Path(first.outputs["path"])
    lock = dest.parent / ".export.lock"
    lock.write_bytes(b"unknown owner")
    before = dest.read_bytes()
    source.write_bytes(b"new payload")
    with pytest.raises(GraphError):
        run()
    assert lock.read_bytes() == b"unknown owner"
    assert dest.read_bytes() == before


def test_external_change_before_publish_is_preserved(setup_export, monkeypatch):
    from studio.nodes import export

    source, _, _, run = setup_export
    first = run()
    dest = Path(first.outputs["path"])
    copy = export.shutil.copy2

    def external_change(src, dst):
        result = copy(src, dst)
        if Path(src) == source:
            dest.write_bytes(b"external new version")
        return result

    source.write_bytes(b"new source")
    monkeypatch.setattr(export.shutil, "copy2", external_change)
    with pytest.raises(GraphError, match="changed before publication"):
        run()
    assert dest.read_bytes() == b"external new version"


def test_installed_identity_read_failure_retains_recovery(setup_export, monkeypatch):
    from studio.nodes import export

    source, _, _, run = setup_export
    first = run()
    dest = Path(first.outputs["path"])
    receipt = Path(first.outputs["manifest"]["manifest_path"])
    old_video, old_receipt = dest.read_bytes(), receipt.read_bytes()
    source.write_bytes(b"new staged bytes")
    replace, stat = export.os.replace, Path.stat
    installed = False

    def install(src, dst):
        nonlocal installed
        result = replace(src, dst)
        if Path(dst) == dest:
            installed = True
        return result

    def unreadable(path, *args, **kwargs):
        if path == dest and installed:
            raise OSError("injected installed identity read failure")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(export.os, "replace", install)
    with monkeypatch.context() as context:
        context.setattr(Path, "stat", unreadable)
        with pytest.raises(GraphError, match="recovery retained"):
            run()
    (recovery,) = dest.parent.glob(".export-*")
    assert (recovery / "previous-video").read_bytes() == old_video
    assert (recovery / "previous-manifest").read_bytes() == old_receipt
    assert receipt.read_bytes() == old_receipt
    assert not (dest.parent / ".export.lock").exists()


@pytest.mark.parametrize("failure", ["write", "close", "rmtree", "replacement"])
def test_lock_failure_cleanup_and_retry(setup_export, monkeypatch, failure):
    from studio.nodes import export

    source, _, _, run = setup_export
    first = run()
    dest = Path(first.outputs["path"])
    receipt = Path(first.outputs["manifest"]["manifest_path"])
    old = dest.read_bytes(), receipt.read_bytes()
    source.write_bytes(b"new payload after lock failure")
    lock = dest.parent / ".export.lock"
    original_open, original_copy = Path.open, export.shutil.copy2

    class FaultyHandle:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def write(self, value):
            if failure == "write":
                self.handle.write(value[:3])
                raise OSError("injected lock write failure")
            return self.handle.write(value)

        def close(self):
            self.handle.close()
            if failure == "close":
                raise OSError("injected lock close failure")

    def faulty_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == lock and args and args[0] == "x":
            return FaultyHandle(handle)
        return handle

    def failed_copy(src, dst):
        if Path(src) == source:
            if failure == "replacement":
                lock.rename(lock.with_name("old-owner-lock"))
                lock.write_bytes(b"unknown replacement owner")
            raise OSError("injected primary copy failure")
        return original_copy(src, dst)

    def failed_cleanup(path):
        raise OSError("injected staging cleanup failure")

    with monkeypatch.context() as context:
        context.setattr(Path, "open", faulty_open)
        if failure in {"rmtree", "replacement"}:
            context.setattr(export.shutil, "copy2", failed_copy)
        if failure == "rmtree":
            context.setattr(export.shutil, "rmtree", failed_cleanup)
        with pytest.raises(GraphError, match="primary copy" if failure in {"rmtree", "replacement"} else "lock"):
            run()
    assert (dest.read_bytes(), receipt.read_bytes()) == old
    if failure == "replacement":
        assert lock.read_bytes() == b"unknown replacement owner"
    else:
        assert not lock.exists()
        assert_receipt(run(), source)
