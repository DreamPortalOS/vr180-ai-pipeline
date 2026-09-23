"""Full production template: script → storyboard stills → review → video → concat → audio → dual export."""

from __future__ import annotations

from studio.models import EdgeSpec, NodeSpec, Project


def production_pipeline_project() -> Project:
    """Correct creative order with a still-review gate before any paid video work."""
    return Project(
        name="production-script-to-dual-export",
        nodes=[
            NodeSpec(
                id="n_brief",
                type="script.project",
                pos=(20, 180),
                params={
                    "title": "红峡谷穿越",
                    "theme": "red rock canyon FPV",
                    "style": "cinematic golden hour, photoreal, full-frame focus, continuous forward motion",
                    "target": "dual",
                    "total_seconds": 12,
                    "aspect_ratio": "1:1",
                    "negative": "cuts, shake, shallow DOF, text, watermark",
                },
            ),
            NodeSpec(
                id="n_shots",
                type="script.shot_list",
                pos=(250, 120),
                params={
                    "shot_texts": "wide establishing over canyon rim\nslow push into the gorge\nemerge into sunlit river bend",
                    "shot_durations": "4,4,4",
                    "motions": "dolly_in,dolly_in,dolly_in",
                },
            ),
            NodeSpec(
                id="n_polish",
                type="text.polish_shots",
                pos=(480, 40),
                params={"provider": "mock", "target": "vr180"},
            ),
            NodeSpec(
                id="n_stills",
                type="image.batch_stills",
                pos=(710, 40),
                params={"provider": "mock", "width": 320, "height": 320},
            ),
            NodeSpec(
                id="n_review",
                type="checkpoint.review",
                pos=(940, 40),
                params={"require_ack": False, "ack": True},
            ),
            NodeSpec(
                id="n_clips",
                type="video.from_stills",
                pos=(710, 280),
                params={"provider": "mock", "fps": 12, "size": 256},
            ),
            NodeSpec(
                id="n_concat",
                type="video.concat",
                pos=(940, 280),
                params={"mode": "concat_demuxer", "filename": "assembled.mp4"},
            ),
            NodeSpec(
                id="n_bgm",
                type="audio.bgm_tone",
                pos=(940, 440),
                params={"duration": 12, "freq_hz": 196, "amplitude": 0.12},
            ),
            NodeSpec(
                id="n_mux",
                type="audio.mux",
                pos=(1170, 280),
                params={"filename": "with_bgm.mp4", "volume": 0.8, "mode": "replace"},
            ),
            NodeSpec(
                id="n_qa",
                type="qa.source_quality",
                pos=(1170, 120),
                params={"mode": "mock"},
            ),
            NodeSpec(
                id="n_dome",
                type="convert.dome",
                pos=(20, 420),
                params={"size": 128, "coverage_h": 150},
            ),
            NodeSpec(
                id="n_cov",
                type="qa.dome_coverage",
                pos=(250, 420),
                params={"min_deg": 70},
            ),
            NodeSpec(
                id="n_vr",
                type="convert.vr180",
                pos=(480, 420),
                params={"mode": "mock", "eye_size": 256},
            ),
            NodeSpec(
                id="n_export",
                type="export.bundle",
                pos=(710, 420),
                params={"filename": "final_master.mp4", "subdir": "studio_export"},
            ),
        ],
        edges=[
            EdgeSpec(id="e1", from_node="n_brief", from_port="brief", to_node="n_shots", to_port="brief"),
            EdgeSpec(id="e2", from_node="n_brief", from_port="style", to_node="n_shots", to_port="style"),
            EdgeSpec(id="e3", from_node="n_shots", from_port="shots", to_node="n_polish", to_port="shots"),
            EdgeSpec(id="e4", from_node="n_polish", from_port="shots", to_node="n_stills", to_port="shots"),
            EdgeSpec(id="e5", from_node="n_stills", from_port="stills", to_node="n_review", to_port="stills"),
            EdgeSpec(id="e6", from_node="n_stills", from_port="sheet", to_node="n_review", to_port="sheet"),
            EdgeSpec(id="e7", from_node="n_review", from_port="stills", to_node="n_clips", to_port="stills"),
            EdgeSpec(id="e8", from_node="n_clips", from_port="videos", to_node="n_concat", to_port="videos"),
            EdgeSpec(id="e9", from_node="n_shots", from_port="total_seconds", to_node="n_bgm", to_port="duration"),
            EdgeSpec(id="e10", from_node="n_concat", from_port="video", to_node="n_mux", to_port="video"),
            EdgeSpec(id="e11", from_node="n_bgm", from_port="audio", to_node="n_mux", to_port="audio"),
            EdgeSpec(id="e12", from_node="n_mux", from_port="video", to_node="n_qa", to_port="video"),
            EdgeSpec(id="e13", from_node="n_mux", from_port="video", to_node="n_dome", to_port="video"),
            EdgeSpec(id="e14", from_node="n_dome", from_port="video", to_node="n_cov", to_port="video"),
            EdgeSpec(id="e15", from_node="n_mux", from_port="video", to_node="n_vr", to_port="video"),
            EdgeSpec(id="e16", from_node="n_vr", from_port="video", to_node="n_export", to_port="video"),
            EdgeSpec(id="e17", from_node="n_brief", from_port="style", to_node="n_export", to_port="prompt"),
        ],
        settings={
            "workflow": "script→stills→review→video→concat→audio→dual-export",
            "default_video_provider": "mock",
            "export": {"vr180": True, "dome": True},
        },
    )
