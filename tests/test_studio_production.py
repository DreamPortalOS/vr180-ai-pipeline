"""End-to-end tests for the production workflow template and new nodes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.graph import list_node_types, run_graph
from studio.nodes.production import (
    AudioMuxNode,
    BatchStillNode,
    BgmToneNode,
    ConcatVideosNode,
    PolishShotsNode,
    ProjectBriefNode,
    ShotListNode,
    VideosFromStillsNode,
)
from studio.templates import production_pipeline_project


def test_production_types_registered() -> None:
    types = {t["type"] for t in list_node_types()}
    assert {
        "script.project",
        "script.shot_list",
        "text.polish_shots",
        "image.batch_stills",
        "checkpoint.review",
        "video.from_stills",
        "video.concat",
        "audio.bgm_tone",
        "audio.mux",
    } <= types


def test_brief_and_shot_list() -> None:
    brief = ProjectBriefNode().run(
        params={"title": "T", "theme": "canyon", "style": "cinematic", "total_seconds": 9},
        inputs={},
        work_dir=".",
        node_id="b",
    )
    shots = ShotListNode().run(
        params={
            "shot_texts": "a\nb\nc",
            "shot_durations": "3,3,3",
            "motions": "dolly_in,dolly_in,dolly_in",
        },
        inputs={"brief": brief["brief"], "style": brief["style"]},
        work_dir=".",
        node_id="s",
    )
    assert shots["summary"]["count"] == 3
    assert shots["total_seconds"] == 9
    assert len(shots["shots"]["shots"]) == 3
    assert "canyon" in shots["shots"]["shots"][0]["prompt"]


def test_polish_and_stills(tmp_path) -> None:
    brief = ProjectBriefNode().run(params={"theme": "valley"}, inputs={}, work_dir=str(tmp_path), node_id="b")
    shot_out = ShotListNode().run(
        params={"shot_texts": "one\ntwo", "shot_durations": "2,2"},
        inputs={"brief": brief["brief"], "style": brief["style"]},
        work_dir=str(tmp_path),
        node_id="s",
    )
    polished = PolishShotsNode().run(
        params={"provider": "mock"},
        inputs={"shots": shot_out["shots"]},
        work_dir=str(tmp_path),
        node_id="p",
    )
    assert polished["meta"]["count"] == 2
    assert polished["shots"]["shots"][0]["prompt"] != shot_out["shots"]["shots"][0]["prompt"]

    stills = BatchStillNode().run(
        params={"provider": "mock", "width": 96, "height": 96},
        inputs={"shots": polished["shots"]},
        work_dir=str(tmp_path),
        node_id="img",
    )
    assert stills["meta"]["count"] == 2
    for s in stills["stills"]["shots"]:
        assert Path(s["image"]).is_file()
    assert Path(stills["sheet"]).is_file()


def test_clips_concat_bgm_mux(tmp_path) -> None:
    brief = ProjectBriefNode().run(params={"theme": "x"}, inputs={}, work_dir=str(tmp_path), node_id="b")
    shot_out = ShotListNode().run(
        params={"shot_texts": "s1\ns2", "shot_durations": "1,1"},
        inputs={"brief": brief["brief"]},
        work_dir=str(tmp_path),
        node_id="s",
    )
    stills = BatchStillNode().run(
        params={"width": 96, "height": 96},
        inputs={"shots": shot_out["shots"]},
        work_dir=str(tmp_path),
        node_id="img",
    )
    clips = VideosFromStillsNode().run(
        params={"size": 96, "fps": 10},
        inputs={"stills": stills["stills"]},
        work_dir=str(tmp_path),
        node_id="v",
    )
    assert clips["meta"]["count"] == 2
    concat = ConcatVideosNode().run(
        params={"filename": "asm.mp4"},
        inputs={"videos": clips["videos"]},
        work_dir=str(tmp_path),
        node_id="c",
    )
    assert Path(concat["video"]).is_file()
    assert concat["meta"]["inputs"] == 2

    bgm = BgmToneNode().run(
        params={"duration": 2, "freq_hz": 200},
        inputs={"duration": 2},
        work_dir=str(tmp_path),
        node_id="a",
    )
    assert Path(bgm["audio"]).is_file()
    mux = AudioMuxNode().run(
        params={"filename": "muxed.mp4"},
        inputs={"video": concat["video"], "audio": bgm["audio"]},
        work_dir=str(tmp_path),
        node_id="m",
    )
    assert Path(mux["video"]).is_file()


def test_review_gate_blocks_when_required(tmp_path) -> None:
    from studio.nodes.production import ReviewGateNode

    payload = {"shots": [{"id": "s1", "image": str(tmp_path / "x.png")}], "summary": {}}
    (tmp_path / "x.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="审核门"):
        ReviewGateNode().run(
            params={"require_ack": True, "ack": False},
            inputs={"stills": payload},
            work_dir=str(tmp_path),
            node_id="r",
        )
    ok = ReviewGateNode().run(
        params={"require_ack": True, "ack": True},
        inputs={"stills": payload},
        work_dir=str(tmp_path),
        node_id="r",
    )
    assert ok["report"]["ack"] is True


def test_full_production_graph(tmp_path) -> None:
    project = production_pipeline_project()
    # shrink for CI speed
    for node in project.nodes:
        if node.type == "image.batch_stills":
            node.params["width"] = 96
            node.params["height"] = 96
        if node.type == "video.from_stills":
            node.params["size"] = 96
            node.params["fps"] = 10
        if node.type == "script.shot_list":
            node.params["shot_texts"] = "open\npush\nbend"
            node.params["shot_durations"] = "1,1,1"
        if node.type == "audio.bgm_tone":
            node.params["duration"] = 3
        if node.type == "convert.dome":
            node.params["size"] = 96

    report = run_graph(project, work_dir=str(tmp_path))
    expected_ok = [
        "n_brief",
        "n_shots",
        "n_polish",
        "n_stills",
        "n_review",
        "n_clips",
        "n_concat",
        "n_bgm",
        "n_mux",
        "n_qa",
        "n_dome",
        "n_cov",
        "n_vr",
        "n_export",
        "n_export_dome",
    ]
    for nid in expected_ok:
        assert report.results[nid].status == "ok", f"{nid}: {report.results[nid].error}"

    sheet = report.results["n_stills"].outputs["sheet"]
    assert Path(sheet).is_file()
    assembled = report.results["n_concat"].outputs["video"]
    assert Path(assembled).is_file()
    muxed = report.results["n_mux"].outputs["video"]
    assert Path(muxed).is_file()
    export = report.results["n_export"].outputs["path"]
    assert Path(export).is_file()
    manifest = json.loads(
        Path(report.results["n_export"].outputs["manifest"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["export_path"] == export

    # PRD §9 dual export: the dome route has its own node, file and manifest.
    dome_export = report.results["n_export_dome"].outputs["path"]
    assert Path(dome_export).is_file()
    dome_manifest = json.loads(
        Path(report.results["n_export_dome"].outputs["manifest"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert dome_manifest["export_path"] == dome_export

    assert export != dome_export
    assert Path(export).name == "final_master.mp4"
    assert Path(dome_export).name == "final_dome.mp4"
    assert Path(export).parent.name == "n_export"
    assert Path(dome_export).parent.name == "n_export_dome"
    assert Path(export).parent.parent.name == "studio_export"
    assert Path(dome_export).parent.parent.name == "studio_export"
    assert Path(export).read_bytes() != Path(dome_export).read_bytes()

    # Each manifest names the converter output the bundle was copied from. The
    # VR180 node runs mode=mock here (a passthrough stand-in, NOT genuine
    # stereo), so this checks routing and byte fidelity only.
    vr_source = report.results["n_vr"].outputs["video"]
    dome_source = report.results["n_cov"].outputs["video"]
    assert manifest["source_video"] == vr_source
    assert dome_manifest["source_video"] == dome_source
    # The dome route exports the video produced by the dome converter.
    assert dome_source == report.results["n_dome"].outputs["video"]
    assert Path(export).read_bytes() == Path(vr_source).read_bytes()
    assert Path(dome_export).read_bytes() == Path(dome_source).read_bytes()

    # coverage report present
    cov = report.results["n_cov"].outputs["report"]
    assert "coverage_deg" in cov
    assert "level" in cov


def test_production_template_wires_both_export_routes() -> None:
    """The template must feed both export nodes, not just VR180 (PRD §9)."""
    project = production_pipeline_project()
    nodes = {n.id: n for n in project.nodes}
    assert nodes["n_export"].params["filename"] == "final_master.mp4"
    assert nodes["n_export_dome"].type == "export.bundle"
    assert nodes["n_export_dome"].params == {"filename": "final_dome.mp4", "subdir": "studio_export"}
    assert nodes["n_export_dome"].pos != nodes["n_export"].pos

    edges = {(e.from_node, e.from_port, e.to_node, e.to_port) for e in project.edges}
    assert ("n_cov", "video", "n_export_dome", "video") in edges
    assert ("n_brief", "style", "n_export_dome", "prompt") in edges
    # The original VR180 route is unchanged.
    assert ("n_vr", "video", "n_export", "video") in edges
    assert ("n_brief", "style", "n_export", "prompt") in edges


def test_server_production_template(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from studio.server import create_app

    client = TestClient(create_app(default_work_dir=str(tmp_path / "w")))
    res = client.get("/api/templates/production")
    assert res.status_code == 200
    body = res.json()
    types = {n["type"] for n in body["nodes"]}
    assert "script.shot_list" in types
    assert "video.concat" in types
    assert "audio.mux" in types

    # shrink and run via API
    for n in body["nodes"]:
        if n["type"] == "image.batch_stills":
            n["params"]["width"] = 96
            n["params"]["height"] = 96
        if n["type"] == "video.from_stills":
            n["params"]["size"] = 96
        if n["type"] == "script.shot_list":
            n["params"]["shot_texts"] = "a\nb"
            n["params"]["shot_durations"] = "1,1"
        if n["type"] == "convert.dome":
            n["params"]["size"] = 96
        if n["type"] == "audio.bgm_tone":
            n["params"]["duration"] = 2

    res = client.post("/api/run", json={"project": body, "work_dir": str(tmp_path / "out")})
    assert res.status_code == 200, res.text
    results = res.json()["results"]
    assert results["n_export"]["status"] == "ok"
    assert results["n_mux"]["status"] == "ok"
