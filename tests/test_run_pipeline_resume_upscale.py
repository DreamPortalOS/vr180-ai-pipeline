"""W-6 (#317): a resume that skips SeedVR2 must adopt the recorded upscale output.

The R-1 pre-stage's entire product is the rewrite it performs on
``args.input``: ``run_seedvr2_prestage`` returns the upscaled clip and
``main()`` points ``args.input`` at it, so depth / stereo / project / encode
all read the higher-resolution file.

The manifest-resume branch used to only log ``"Skipping SeedVR2 pre-stage"``
and leave ``args.input`` alone.  Every downstream stage then ran on the
ORIGINAL, un-upscaled footage while the manifest claimed ``upscale`` was done
— a silent quality downgrade with a fully green log.  The correct path was
already sitting in the manifest: :func:`_stage_artifacts` records
``outputs = [args.input]`` for ``upscale`` *after* the rewrite.

Semantics pinned here (also stated in ``--resume-from --help`` and in
``_resume_upscaled_input``'s docstring):

* recorded output present  → ``args.input`` becomes it, pre-stage NOT re-run;
* recorded output missing / not recorded at all → **abort, exit 1**.  Never a
  silent fall-through to the original, and never an implicit (expensive)
  re-upscale.

Mutation check the card asks for: drop ``args.input = _resume_upscaled_input(
manifest)`` from the skip branch in ``main()`` and every test in
``TestResumeAdoptsTheRecordedUpscale`` plus the two ``missing`` cases go red.

Everything heavy is mocked — no ffmpeg, no models, no real video decode — and
every path handed to ``main()`` lives under ``tmp_path``, so this suite never
writes into the repo.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.job_manifest import mark_stage_done, new_manifest, save_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _import_run_pipeline():
    """Load scripts/run_pipeline.py as an isolated module (V-4 test convention)."""
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        name = f"run_pipeline_rsu{os.getpid()}_{id(__file__)}"
        spec = importlib.util.spec_from_file_location(
            name,
            os.path.join(scripts_dir, "run_pipeline.py"),
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(scripts_dir)


@pytest.fixture(scope="module")
def run_pipeline():
    return _import_run_pipeline()


# --------------------------------------------------------------------------
# fixtures: a real source file + a real manifest, both under tmp_path
# --------------------------------------------------------------------------


def _make_source(tmp_path: Path) -> Path:
    """The operator's footage.  Must really exist: ``validate_source`` hashes it."""
    src = tmp_path / "in.mp4"
    src.write_bytes(b"original-480p-footage")
    return src


def _make_upscaled(tmp_path: Path) -> Path:
    """The SeedVR2 intermediate the first run produced."""
    up = tmp_path / "temp" / "in_seedvr2_2x.mp4"
    up.parent.mkdir(parents=True, exist_ok=True)
    up.write_bytes(b"upscaled-2x-footage-bigger")
    return up


def _write_manifest(tmp_path: Path, source: Path, outputs: list[str]) -> Path:
    """Manifest with ``upscale`` marked done and *outputs* recorded.

    ``mark_stage_done`` hashes the outputs that exist at record time
    (``hash_paths`` silently omits missing files), which is what lets the
    "recorded but absent" case reach our guard rather than tripping
    ``validate_stage_outputs`` first.
    """
    manifest = new_manifest("job-317", source)
    mark_stage_done(
        manifest,
        "upscale",
        machine="win-cuda",
        inputs=[],
        outputs=outputs,
        params={"video_upscale": "seedvr2", "video_upscale_factor": 2, "upscale": 0},
    )
    path = tmp_path / "job.json"
    save_manifest(manifest, path)
    return path


def _main_args(tmp_path: Path, source: Path, **overrides):
    """Args for a minimal batch run through ``main()``.

    Every filesystem-facing attribute points into ``tmp_path``: a MagicMock
    left on ``temp_dir`` would make ``get_temp_dir`` mkdir a literal
    ``MagicMock/`` tree in the repo root.
    """
    args = MagicMock()
    args.inputs = None
    args.input = str(source)
    args.output = str(tmp_path / "out.mp4")
    args.temp_dir = str(tmp_path / "temp")
    args.keep_temp = False
    args.video_upscale = "seedvr2"
    args.video_upscale_factor = 2
    args.upscale = 0
    args.device = "cpu"
    args.validate_input = False
    args.fps = 30
    # --resume-from forces the batch path anyway (a manifest run cannot be a
    # single fused stream); False keeps that explicit.
    args.streaming = False
    args.stage = "all"
    args.force_sbs = False
    args.projection = "vr180"
    args.input_projection = "rectilinear"
    args.model_size = "small"
    args.ipd = 0.064
    args.max_disparity = 0.05
    args.output_width = 2880
    args.output_height = 2880
    args.src_hfov = 70.0
    args.codec = "h264"
    args.crf = 23
    args.bitrate = "45M"
    args.max_frames = None
    args.comfort = "balanced"
    args.convergence = None
    args.no_temporal = False
    args.preset = "source"
    args.gop = None
    args.depth_model = "depth-anything"
    args.stereo_model = "default"
    args.copy_audio_from = None
    args.manifest = None
    args.stages = None
    args.resume_from = None
    # Both gates are other cards' business (#311/#314, #230); 'off' keeps this
    # suite about the resume rewrite only.
    args.preflight = "off"
    args.source_check = "off"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@contextlib.contextmanager
def _mocked_main(run_pipeline, args):
    """Run ``main()`` with every expensive site mocked; yield the mock dict."""
    mocks: dict[str, MagicMock] = {}
    with contextlib.ExitStack() as stack:
        for name in ("run_seedvr2_prestage", "SeedVR2Upscaler", "_stage_all_body", "detect_sbs_input"):
            mocks[name] = stack.enter_context(patch.object(run_pipeline, name))
        # The fresh (non-resume) path rewrites args.input to this; a resumed
        # run must never produce it, because the pre-stage must not run.
        mocks["run_seedvr2_prestage"].return_value = "freshly_upscaled_2x.mp4"
        mocks["detect_sbs_input"].return_value = False
        stack.enter_context(patch.object(run_pipeline, "parse_args", return_value=args))
        stack.enter_context(patch.object(run_pipeline, "apply_quality_preset"))
        stack.enter_context(patch.object(run_pipeline, "validate_input_projection"))
        stack.enter_context(patch.object(run_pipeline, "_run_preflight"))
        yield mocks


class TestResumeAdoptsTheRecordedUpscale:
    """Manifest says ``upscale`` done and the artefact is there."""

    # Acceptance: args.input points at the upscaled file AND the pre-stage is
    # never called (the whole point of resuming is not paying for it twice).
    def test_input_becomes_the_recorded_output_and_prestage_never_runs(self, run_pipeline, tmp_path):
        source = _make_source(tmp_path)
        upscaled = _make_upscaled(tmp_path)
        manifest_path = _write_manifest(tmp_path, source, [str(upscaled)])
        args = _main_args(tmp_path, source, resume_from=str(manifest_path))

        with _mocked_main(run_pipeline, args) as mocks:
            run_pipeline.main()

        assert args.input == str(upscaled)
        assert mocks["run_seedvr2_prestage"].call_count == 0
        assert mocks["SeedVR2Upscaler"].call_count == 0

    def test_downstream_stages_see_the_upscaled_clip_not_the_original(self, run_pipeline, tmp_path):
        """The regression this card exists for: everything below the pre-stage
        must read the upscaled file.  ``detect_sbs_input`` is the first
        consumer of ``args.input`` after the branch, so it is the honest
        witness for "what did the rest of the run actually get?"."""
        source = _make_source(tmp_path)
        upscaled = _make_upscaled(tmp_path)
        manifest_path = _write_manifest(tmp_path, source, [str(upscaled)])
        args = _main_args(tmp_path, source, resume_from=str(manifest_path))

        seen: list[str] = []
        with _mocked_main(run_pipeline, args) as mocks:
            mocks["detect_sbs_input"].side_effect = lambda path, **kw: seen.append(path) or False
            mocks["_stage_all_body"].side_effect = lambda a, *rest: seen.append(a.input)
            run_pipeline.main()

        assert seen == [str(upscaled), str(upscaled)]
        assert str(source) not in seen

    def test_log_names_the_path_actually_resumed_from(self, run_pipeline, tmp_path, caplog):
        """Aligned with the fresh path's ``input replaced A → B`` line, so the
        operator can see which file the run continued from."""
        caplog.set_level(run_pipeline.logging.INFO)
        source = _make_source(tmp_path)
        upscaled = _make_upscaled(tmp_path)
        manifest_path = _write_manifest(tmp_path, source, [str(upscaled)])
        args = _main_args(tmp_path, source, resume_from=str(manifest_path))

        with _mocked_main(run_pipeline, args):
            run_pipeline.main()

        assert f"input replaced {source} → {upscaled}" in caplog.text

    def test_manifest_entry_is_left_alone(self, run_pipeline, tmp_path):
        """Adopting the artefact must not rewrite the stage record — a resumed
        run that re-saves the manifest has to keep the original output path,
        not re-point it at itself."""
        source = _make_source(tmp_path)
        upscaled = _make_upscaled(tmp_path)
        manifest_path = _write_manifest(tmp_path, source, [str(upscaled)])
        out_manifest = tmp_path / "job-out.json"
        args = _main_args(
            tmp_path,
            source,
            resume_from=str(manifest_path),
            manifest=str(out_manifest),
        )

        with _mocked_main(run_pipeline, args) as mocks:
            # _stage_all_body owns the manifest save; run the real save here so
            # the assertion is about persisted state, not an in-memory dict.
            def _save(a, temp_dir, is_sbs, manifest, skip, stages, touched):
                save_manifest(manifest, out_manifest)

            mocks["_stage_all_body"].side_effect = _save
            run_pipeline.main()

        data = json.loads(out_manifest.read_text(encoding="utf-8"))
        upscale_stage = next(s for s in data["stages"] if s["name"] == "upscale")
        assert upscale_stage["outputs"] == [str(upscaled)]
        assert upscale_stage["status"] == "done"


class TestMissingArtefactAborts:
    """No usable recorded output → exit 1.  Never silent, never a re-upscale."""

    # Acceptance: the recorded file is gone (hashes empty, so
    # validate_stage_outputs passes vacuously and our guard is what fires).
    def test_recorded_output_absent_exits_one(self, run_pipeline, tmp_path, caplog):
        caplog.set_level(run_pipeline.logging.ERROR)
        source = _make_source(tmp_path)
        gone = tmp_path / "temp" / "in_seedvr2_2x.mp4"  # never created
        manifest_path = _write_manifest(tmp_path, source, [str(gone)])
        args = _main_args(tmp_path, source, resume_from=str(manifest_path))

        with _mocked_main(run_pipeline, args) as mocks, pytest.raises(SystemExit) as exc_info:
            run_pipeline.main()

        assert exc_info.value.code == 1
        # Not a silent downgrade...
        assert args.input == str(source)
        assert mocks["_stage_all_body"].call_count == 0
        # ...and not a silent (expensive) re-upscale either.
        assert mocks["run_seedvr2_prestage"].call_count == 0
        assert str(gone) in caplog.text
        assert "un-upscaled" in caplog.text

    def test_no_output_recorded_exits_one(self, run_pipeline, tmp_path, caplog):
        """Reachable for real: a first run with ``--upscale N --video-upscale
        none`` marks ``upscale`` done with ``outputs = []`` (see
        ``_stage_artifacts``).  Resuming that manifest with ``--video-upscale
        seedvr2`` has nothing to adopt."""
        caplog.set_level(run_pipeline.logging.ERROR)
        source = _make_source(tmp_path)
        manifest_path = _write_manifest(tmp_path, source, [])
        args = _main_args(tmp_path, source, resume_from=str(manifest_path))

        with _mocked_main(run_pipeline, args) as mocks, pytest.raises(SystemExit) as exc_info:
            run_pipeline.main()

        assert exc_info.value.code == 1
        assert args.input == str(source)
        assert mocks["_stage_all_body"].call_count == 0
        assert mocks["run_seedvr2_prestage"].call_count == 0
        assert "records no output path" in caplog.text

    def test_hashed_output_deleted_is_rejected_by_manifest_validation(self, run_pipeline, tmp_path):
        """The other half of the same contract: when the output *was* hashed at
        record time, ``validate_stage_outputs`` rejects the resume before we
        get here.  Either way the run stops instead of downgrading."""
        source = _make_source(tmp_path)
        upscaled = _make_upscaled(tmp_path)
        manifest_path = _write_manifest(tmp_path, source, [str(upscaled)])
        upscaled.unlink()
        args = _main_args(tmp_path, source, resume_from=str(manifest_path))

        with _mocked_main(run_pipeline, args) as mocks, pytest.raises(SystemExit) as exc_info:
            run_pipeline.main()

        assert exc_info.value.code == 1
        assert args.input == str(source)
        assert mocks["_stage_all_body"].call_count == 0
        assert mocks["run_seedvr2_prestage"].call_count == 0


class TestUnrelatedPathsUnchanged:
    """The fix must be confined to the manifest-skip branch."""

    def test_fresh_run_still_upscales_and_rewrites(self, run_pipeline, tmp_path):
        source = _make_source(tmp_path)
        args = _main_args(tmp_path, source)  # no --resume-from

        with _mocked_main(run_pipeline, args) as mocks:
            run_pipeline.main()

        assert mocks["run_seedvr2_prestage"].call_count == 1
        assert args.input == "freshly_upscaled_2x.mp4"

    def test_resume_without_seedvr2_never_consults_the_upscale_record(self, run_pipeline, tmp_path):
        """``--video-upscale none``: there is no pre-stage to skip, so a
        manifest whose ``upscale`` output is long gone must not abort the run."""
        source = _make_source(tmp_path)
        gone = tmp_path / "temp" / "in_seedvr2_2x.mp4"  # never created
        manifest_path = _write_manifest(tmp_path, source, [str(gone)])
        args = _main_args(
            tmp_path,
            source,
            video_upscale="none",
            resume_from=str(manifest_path),
        )

        with _mocked_main(run_pipeline, args) as mocks:
            run_pipeline.main()

        assert args.input == str(source)
        assert mocks["_stage_all_body"].call_count == 1
        assert mocks["run_seedvr2_prestage"].call_count == 0


class TestHelpDocumentsTheSemantics:
    def test_resume_from_help_states_the_adopt_and_abort_rules(self, run_pipeline):
        flat = " ".join(run_pipeline._build_parser().format_help().split())
        assert "downstream stages continue from the UPSCALED clip, not the original" in flat
        assert "the run ABORTS (exit 1) instead of silently continuing at the original resolution" in flat
