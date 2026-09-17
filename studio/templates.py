"""Demo project with dual VR180 + dome export path (mock providers)."""

from __future__ import annotations

from studio.models import EdgeSpec, NodeSpec, Project


def dual_export_demo_project() -> Project:
    """Canvas template laid out to fit a typical laptop viewport without scrolling."""
    return Project(
        name="demo-dual-export",
        nodes=[
            NodeSpec(
                id="n_script",
                type="script.storyboard",
                pos=(24, 160),
                params={
                    "title": "dual route demo",
                    "prompt": "FPV drone flying slowly through a red canyon at golden hour",
                    "duration": 2,
                    "aspect_ratio": "1:1",
                },
            ),
            NodeSpec(
                id="n_polish",
                type="text.llm_polish",
                pos=(260, 40),
                params={"provider": "mock", "target": "vr180"},
            ),
            NodeSpec(
                id="n_vid",
                type="video.seedance",
                pos=(500, 160),
                params={"provider": "mock", "size": 96, "ratio": "1:1"},
            ),
            NodeSpec(
                id="n_qa",
                type="qa.source_quality",
                pos=(740, 40),
                params={"mode": "mock"},
            ),
            NodeSpec(
                id="n_dome",
                type="convert.dome",
                pos=(260, 320),
                params={"size": 128, "coverage_h": 150},
            ),
            NodeSpec(
                id="n_cov",
                type="qa.dome_coverage",
                pos=(500, 320),
                params={"min_deg": 70},
            ),
            NodeSpec(
                id="n_vr",
                type="convert.vr180",
                pos=(740, 320),
                params={"mode": "mock"},
            ),
            NodeSpec(
                id="n_exp",
                type="export.bundle",
                pos=(980, 160),
                params={"filename": "dual_demo.mp4"},
            ),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="n_script", from_port="prompt", to_node="n_polish", to_port="prompt"),
            EdgeSpec(id="e2", from_node="n_polish", from_port="prompt", to_node="n_vid", to_port="prompt"),
            EdgeSpec(id="e3", from_node="n_script", from_port="duration", to_node="n_vid", to_port="duration"),
            EdgeSpec(id="e4", from_node="n_vid", from_port="video", to_node="n_qa", to_port="video"),
            EdgeSpec(id="e5", from_node="n_qa", from_port="video", to_node="n_dome", to_port="video"),
            EdgeSpec(id="e6", from_node="n_dome", from_port="video", to_node="n_cov", to_port="video"),
            EdgeSpec(id="e7", from_node="n_qa", from_port="video", to_node="n_vr", to_port="video"),
            EdgeSpec(id="e8", from_node="n_vr", from_port="video", to_node="n_exp", to_port="video"),
            EdgeSpec(id="e9", from_node="n_polish", from_port="prompt", to_node="n_exp", to_port="prompt"),
        ],
        settings={
            "default_video_provider": "mock",
            "export": {"vr180": True, "dome": True},
        },
    )
