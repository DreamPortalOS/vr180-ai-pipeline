"""Tests for V-3 job manifest + cross-machine staged pipeline (issue #36).

Covers:
  - pipeline.job_manifest: roundtrip read/write, stage bookkeeping, hashing
  - scripts.run_pipeline: --stages subset parsing, resume skip logic,
    hash-mismatch abort, and zero-change regression without the new flags

All tests are CPU-only and mock every real conversion / model dependency.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.job_manifest import (  # noqa: E402
    STATUS_DONE,
    STATUS_PENDING,
    ManifestError,
    completed_stages,
    get_stage,
    hash_paths,
    load_manifest,
    mark_stage_done,
    new_manifest,
    save_manifest,
    sha256_file,
    validate_source,
    validate_stage_outputs,
)


def _import_run_pipeline():
    """Import scripts/run_pipeline.py as a module (fresh each call)."""
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    sys.path.insert(0, scripts_dir)
    try:
        sys.modules.pop("run_pipeline", None)
        import run_pipeline

        return run_pipeline
    finally:
        sys.path.remove(scripts_dir)


# ---------------------------------------------------------------------------
# pipeline.job_manifest
# ---------------------------------------------------------------------------


class TestManifestRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"fake-video-bytes")
            m = new_manifest("job-1", src)
            m["stages"][0]["params"] = {"factor": 2}
            path = Path(td) / "job.json"
            save_manifest(m, path)

            loaded = load_manifest(path)
            self.assertEqual(loaded["job_id"], "job-1")
            self.assertEqual(loaded["source_hash"], sha256_file(src))
            self.assertEqual(len(loaded["stages"]), 5)
            self.assertEqual(loaded["stages"][0]["params"], {"factor": 2})
            self.assertEqual(loaded, m)

    def test_new_manifest_all_pending(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"x")
            m = new_manifest("job-2", src)
            self.assertEqual(completed_stages(m), [])
            for s in m["stages"]:
                self.assertEqual(s["status"], STATUS_PENDING)

    def test_load_manifest_missing_file(self):
        with self.assertRaises(ManifestError):
            load_manifest("/nonexistent/path/job.json")

    def test_load_manifest_invalid_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_load_manifest_missing_keys(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad2.json"
            p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_manifest(p)


class TestManifestHashing(unittest.TestCase):
    def test_sha256_file_stable(self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.bin"
            p.write_bytes(b"abc" * 1000)
            self.assertEqual(sha256_file(p), hashlib.sha256(b"abc" * 1000).hexdigest())

    def test_hash_paths_skips_dirs_and_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "out.mp4"
            f.write_bytes(b"data")
            d = Path(td) / "frames"
            d.mkdir()
            hashes = hash_paths([str(f), str(d), str(Path(td) / "nope.bin")])
            self.assertEqual(list(hashes), [str(f)])

    def test_mark_stage_done_records_hashes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"src")
            out = Path(td) / "out.mp4"
            out.write_bytes(b"encoded")
            m = new_manifest("job-3", src)
            stage = mark_stage_done(m, "encode", machine="win-cuda", outputs=[str(out)], params={"codec": "h264"})
            self.assertEqual(stage["status"], STATUS_DONE)
            self.assertEqual(stage["machine"], "win-cuda")
            self.assertEqual(stage["hashes"][str(out)], sha256_file(out))
            self.assertEqual(completed_stages(m), ["encode"])
            self.assertIs(get_stage(m, "encode"), stage)


class TestManifestValidation(unittest.TestCase):
    def _manifest_with_done_stage(self, td):
        src = Path(td) / "src.mp4"
        src.write_bytes(b"src")
        out = Path(td) / "out.mp4"
        out.write_bytes(b"artifact")
        m = new_manifest("job-4", src)
        mark_stage_done(m, "depth", outputs=[str(out)])
        return m, src, out

    def test_validate_ok(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            m, _, _ = self._manifest_with_done_stage(td)
            validate_stage_outputs(m, "depth")  # no raise

    def test_validate_hash_mismatch_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            m, _, out = self._manifest_with_done_stage(td)
            out.write_bytes(b"tampered")
            with self.assertRaises(ManifestError) as ctx:
                validate_stage_outputs(m, "depth")
            self.assertIn("hash mismatch", str(ctx.exception))

    def test_validate_missing_output_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            m, _, out = self._manifest_with_done_stage(td)
            out.unlink()
            with self.assertRaises(ManifestError) as ctx:
                validate_stage_outputs(m, "depth")
            self.assertIn("missing", str(ctx.exception))

    def test_validate_pending_stage_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src.mp4"
            src.write_bytes(b"src")
            m = new_manifest("job-5", src)
            with self.assertRaises(ManifestError):
                validate_stage_outputs(m, "stereo")

    def test_validate_source_mismatch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            m, src, _ = self._manifest_with_done_stage(td)
            validate_source(m, src)  # ok
            src.write_bytes(b"different-source")
            with self.assertRaises(ManifestError) as ctx:
                validate_source(m, src)
            self.assertIn("Source hash mismatch", str(ctx.exception))


# ---------------------------------------------------------------------------
# scripts.run_pipeline — --stages parsing
# ---------------------------------------------------------------------------


class TestParseStagesArg(unittest.TestCase):
    def test_none_means_all(self):
        rp = _import_run_pipeline()
        self.assertEqual(rp.parse_stages_arg(None), ["upscale", "depth", "stereo", "project", "encode"])

    def test_subset(self):
        rp = _import_run_pipeline()
        self.assertEqual(rp.parse_stages_arg("depth,stereo"), ["depth", "stereo"])

    def test_canonical_order_and_dedup(self):
        rp = _import_run_pipeline()
        self.assertEqual(rp.parse_stages_arg("encode,depth,depth"), ["depth", "encode"])

    def test_unknown_stage_raises(self):
        rp = _import_run_pipeline()
        with self.assertRaises(ValueError):
            rp.parse_stages_arg("depth,explode")

    def test_cli_accepts_stages(self):
        rp = _import_run_pipeline()
        args = rp.parse_args(["--input", "x.mp4", "--stages", "depth,stereo"])
        self.assertEqual(args.stages, "depth,stereo")
        self.assertIsNone(args.manifest)
        self.assertIsNone(args.resume_from)


# ---------------------------------------------------------------------------
# scripts.run_pipeline — manifest prepare / skip logic
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    base = dict(
        input="input.mp4",
        output=None,
        stage="all",
        stages=None,
        manifest=None,
        resume_from=None,
        machine=None,
        device="cpu",
        video_upscale="none",
        video_upscale_factor=2,
        upscale=0,
        depth_model="depth-anything",
        model_size="small",
        stereo_model="default",
        ipd=0.064,
        output_width=1920,
        output_height=1920,
        src_hfov=70.0,
        outpaint="none",
        codec="h264",
        crf=23,
        bitrate=None,
        fps=30,
        temp_dir=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestManifestPrepare(unittest.TestCase):
    def test_no_flags_returns_none(self):
        rp = _import_run_pipeline()
        self.assertEqual(rp._manifest_prepare(_make_args()), (None, None, None))

    def test_unknown_stages_exits_2(self):
        rp = _import_run_pipeline()
        with self.assertRaises(SystemExit) as ctx:
            rp._manifest_prepare(_make_args(stages="depth,bogus"))
        self.assertEqual(ctx.exception.code, 2)

    def test_new_manifest_created(self):
        import tempfile

        rp = _import_run_pipeline()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            mpath = str(Path(td) / "job.json")
            args = _make_args(input=str(src), manifest=mpath, stages="depth,stereo")
            manifest, skip, stages = rp._manifest_prepare(args)
            self.assertIsNotNone(manifest)
            self.assertEqual(skip, [])
            self.assertEqual(stages, ["depth", "stereo"])
            self.assertEqual(manifest["source_hash"], sha256_file(src))

    def test_resume_skips_done_stages(self):
        import tempfile

        rp = _import_run_pipeline()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            m = new_manifest("job-6", src)
            out = Path(td) / "artifact.bin"
            out.write_bytes(b"a")
            mark_stage_done(m, "upscale", outputs=[str(out)])
            mark_stage_done(m, "depth", outputs=[str(out)])
            mpath = Path(td) / "job.json"
            save_manifest(m, mpath)

            args = _make_args(input=str(src), resume_from=str(mpath))
            _manifest, skip, stages = rp._manifest_prepare(args)
            self.assertEqual(skip, ["upscale", "depth"])
            self.assertEqual(stages, ["stereo", "equirect", "outpaint", "metadata"])

    def test_resume_hash_mismatch_exits_1(self):
        import tempfile

        rp = _import_run_pipeline()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            m = new_manifest("job-7", src)
            out = Path(td) / "artifact.bin"
            out.write_bytes(b"a")
            mark_stage_done(m, "depth", outputs=[str(out)])
            mpath = Path(td) / "job.json"
            save_manifest(m, mpath)
            out.write_bytes(b"tampered")  # corrupt the artifact after manifest write

            args = _make_args(input=str(src), resume_from=str(mpath))
            with self.assertRaises(SystemExit) as ctx:
                rp._manifest_prepare(args)
            self.assertEqual(ctx.exception.code, 1)

    def test_resume_source_mismatch_exits_1(self):
        import tempfile

        rp = _import_run_pipeline()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            m = new_manifest("job-8", src)
            mpath = Path(td) / "job.json"
            save_manifest(m, mpath)
            src.write_bytes(b"other-source")

            args = _make_args(input=str(src), resume_from=str(mpath))
            with self.assertRaises(SystemExit) as ctx:
                rp._manifest_prepare(args)
            self.assertEqual(ctx.exception.code, 1)

    def test_resume_missing_manifest_exits_1(self):
        rp = _import_run_pipeline()
        args = _make_args(resume_from="/nonexistent/job.json")
        with self.assertRaises(SystemExit) as ctx:
            rp._manifest_prepare(args)
        self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# scripts.run_pipeline — main() stage skip / record / regression
# ---------------------------------------------------------------------------


class TestMainManifestFlow(unittest.TestCase):
    """main() with mocked stage functions — verifies skip/record behaviour."""

    def test_resume_skips_done_and_records_rest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            m = new_manifest("job-9", src)
            artifact = Path(td) / "artifact.bin"
            artifact.write_bytes(b"a")
            mark_stage_done(m, "upscale", outputs=[str(artifact)])
            mark_stage_done(m, "depth", outputs=[str(artifact)])
            mpath = Path(td) / "job.json"
            save_manifest(m, mpath)
            out_manifest = str(Path(td) / "job_out.json")

            rp = _import_run_pipeline()
            recorded = {}
            argv = [
                "--input",
                str(src),
                "--resume-from",
                str(mpath),
                "--manifest",
                out_manifest,
                "--device",
                "cpu",
                "--temp-dir",
                td,
            ]
            with (
                patch.object(rp, "apply_quality_preset", lambda a: None),
                patch.object(rp, "detect_sbs_input", return_value=False),
                patch.object(rp, "read_frames", return_value=iter([])),
                patch.object(rp, "run_upscale_stage", side_effect=AssertionError("must skip")),
                patch.object(rp, "run_depth_stage", side_effect=AssertionError("must skip")),
                patch.object(rp, "run_stereo_stage", return_value=([], [])) as m_stereo,
                patch.object(rp, "run_equirect_stage", return_value=[]),
                patch.object(rp, "run_outpaint_stage", return_value=[]),
                patch.object(rp, "run_metadata_stage", return_value="out.mp4"),
                patch.object(rp, "save_checkpoint"),
                patch.object(
                    rp,
                    "_manifest_record_stage",
                    side_effect=lambda man, a, n: recorded.setdefault("stages", []).append(n),
                ),
                patch("cv2.VideoCapture") as m_cap,
            ):
                cap = m_cap.return_value
                cap.get.return_value = 30.0
                cap.isOpened.return_value = True
                with patch.object(sys, "argv", ["run_pipeline.py", *argv]):
                    rp.main()

            m_stereo.assert_called_once()
            # upscale+depth skipped; project records once (equirect), encode once
            self.assertEqual(recorded["stages"], ["stereo", "project", "encode"])
            # Manifest persisted with resumed stages still marked done
            saved = load_manifest(out_manifest)
            self.assertEqual(completed_stages(saved), ["upscale", "depth"])

    def test_stages_subset_runs_only_those(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            rp = _import_run_pipeline()
            argv = ["--input", str(src), "--stages", "depth", "--device", "cpu", "--temp-dir", td]
            with (
                patch.object(rp, "apply_quality_preset", lambda a: None),
                patch.object(rp, "detect_sbs_input", return_value=False),
                patch.object(rp, "read_frames", return_value=iter([])),
                patch.object(rp, "run_depth_stage", return_value=[]) as m_depth,
                patch.object(rp, "run_stereo_stage", side_effect=AssertionError("not selected")),
                patch.object(rp, "run_equirect_stage", side_effect=AssertionError("not selected")),
                patch.object(rp, "run_metadata_stage", side_effect=AssertionError("not selected")),
                patch.object(rp, "save_checkpoint"),
                patch("cv2.VideoCapture") as m_cap,
            ):
                cap = m_cap.return_value
                cap.get.return_value = 30.0
                cap.isOpened.return_value = True
                with patch.object(sys, "argv", ["run_pipeline.py", *argv]):
                    rp.main()
            m_depth.assert_called_once()


class TestRegressionNoNewFlags(unittest.TestCase):
    """不带新参数时行为零变化：所有 stage 照常执行，不做任何 manifest IO。"""

    def test_full_run_without_manifest_flags(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.mp4"
            src.write_bytes(b"src")
            rp = _import_run_pipeline()
            argv = ["--input", str(src), "--device", "cpu", "--temp-dir", td]
            with (
                patch.object(rp, "apply_quality_preset", lambda a: None),
                patch.object(rp, "detect_sbs_input", return_value=False),
                patch.object(rp, "read_frames", return_value=iter([])),
                patch.object(rp, "run_depth_stage", return_value=[]) as m_depth,
                patch.object(rp, "run_stereo_stage", return_value=([], [])) as m_stereo,
                patch.object(rp, "run_equirect_stage", return_value=[]) as m_eq,
                patch.object(rp, "run_outpaint_stage", return_value=[]) as m_out,
                patch.object(rp, "run_metadata_stage", return_value="out.mp4") as m_meta,
                patch.object(rp, "save_checkpoint") as m_ckpt,
                patch.object(rp, "_manifest_record_stage") as m_rec,
                patch("pipeline.job_manifest.save_manifest") as m_save,
                patch("cv2.VideoCapture") as m_cap,
            ):
                cap = m_cap.return_value
                cap.get.return_value = 30.0
                cap.isOpened.return_value = True
                with patch.object(sys, "argv", ["run_pipeline.py", *argv]):
                    rp.main()

            # All stages ran exactly once (upscale off by default)
            m_depth.assert_called_once()
            m_stereo.assert_called_once()
            m_eq.assert_called_once()
            m_out.assert_called_once()
            m_meta.assert_called_once()
            # Checkpoints still written (legacy resume path untouched):
            # depth / stereo / equirect / outpaint (metadata stage never checkpoints)
            self.assertEqual(m_ckpt.call_count, 4)
            # No manifest machinery engaged
            m_rec.assert_not_called()
            m_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
