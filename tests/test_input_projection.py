"""Tests for the --input-projection source paths (equirect #239/#286, fisheye #294)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scripts import run_pipeline as rp


class _Capture:
    def __init__(self, width, height, opened=True):
        self.width = width
        self.height = height
        self.opened = opened

    def isOpened(self):  # noqa: N802 - mirrors cv2.VideoCapture
        return self.opened

    def get(self, prop):
        if prop == rp.cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == rp.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0

    def release(self):
        pass


def test_default_input_projection_preserves_rectilinear_pipeline():
    args = rp.parse_args(["--input", "source.mp4"])

    assert args.input_projection == "rectilinear"
    assert args.src_hfov == 70.0
    assert args._src_hfov_explicit is False
    assert "equirect" in rp.STAGE_ORDER


def test_equirect_input_uses_depth_stereo_metadata_without_projection():
    args = rp.parse_args(["--input", "source.mp4", "--input-projection", "equirect"])

    assert rp.STAGE_ORDER_EQUIRECT == ["upscale", "depth", "stereo", "outpaint", "metadata"]
    assert "equirect" not in rp.STAGE_ORDER_EQUIRECT
    assert "depth" in rp.STAGE_ORDER_EQUIRECT
    assert "stereo" in rp.STAGE_ORDER_EQUIRECT
    assert "metadata" in rp.STAGE_ORDER_EQUIRECT
    assert args._src_hfov_explicit is False


def test_explicit_src_hfov_warns_for_equirect(monkeypatch, caplog):
    monkeypatch.setattr(rp.cv2, "VideoCapture", lambda _: _Capture(1024, 1024))
    args = SimpleNamespace(input="source.mp4", input_projection="equirect", _src_hfov_explicit=True)

    rp.validate_input_projection(args)

    assert "src-hfov" in caplog.text


def test_non_square_equirect_input_warns_with_actual_dimensions(monkeypatch, caplog):
    monkeypatch.setattr(rp.cv2, "VideoCapture", lambda _: _Capture(1920, 1080))
    args = SimpleNamespace(input="source.mp4", input_projection="equirect", _src_hfov_explicit=False)

    rp.validate_input_projection(args)

    assert "1920x1080" in caplog.text
    assert "square" in caplog.text


def test_square_equirect_input_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(rp.cv2, "VideoCapture", lambda _: _Capture(1024, 1024))
    args = SimpleNamespace(input="source.mp4", input_projection="equirect", _src_hfov_explicit=False)

    rp.validate_input_projection(args)

    assert caplog.text == ""


def test_equirect_passthrough_joins_eyes_without_rendering(monkeypatch, tmp_path):
    written = []
    monkeypatch.setattr(rp, "get_temp_dir", lambda args, name: str(tmp_path / name))
    monkeypatch.setattr(rp.cv2, "cvtColor", lambda frame, _code: frame)
    monkeypatch.setattr(rp.cv2, "imwrite", lambda path, frame: written.append((path, frame)) or True)
    args = SimpleNamespace()
    left = [np.zeros((2, 2, 3), dtype=np.uint8)]
    right = [np.ones((2, 2, 3), dtype=np.uint8)]

    result = rp.run_equirect_passthrough_stage(args, left, right)

    assert result[0].shape == (2, 4, 3)
    assert np.array_equal(result[0][:, :2], left[0])
    assert np.array_equal(result[0][:, 2:], right[0])
    assert written[0][0].endswith("equirect_000000.png")


# --------------------------------------------------------------------------- #
# C-2 (#294): --input-projection fisheye
# --------------------------------------------------------------------------- #


def test_fisheye_is_an_accepted_input_projection():
    args = rp.parse_args(["--input", "source.mp4", "--input-projection", "fisheye"])

    assert args.input_projection == "fisheye"
    assert args.fisheye_fov == 180.0  # the inscribed circle covers the hemisphere
    assert args._src_hfov_explicit is False


def test_fisheye_fov_is_configurable():
    args = rp.parse_args(["--input", "source.mp4", "--input-projection", "fisheye", "--fisheye-fov", "150"])

    assert args.fisheye_fov == 150.0


class _MapperConstructedError(Exception):
    """Sentinel so the stage aborts once the mapper kwargs are captured."""


def _capture_mapper_kwargs(monkeypatch):
    captured = {}

    def _recorder(**kwargs):
        captured.update(kwargs)
        raise _MapperConstructedError

    monkeypatch.setattr(rp, "EquirectangularMapper", _recorder)
    return captured


def test_equirect_stage_builds_the_mapper_with_the_fisheye_knobs(monkeypatch):
    """The batch projection stage must hand the source projection to the
    mapper — otherwise a fisheye source is silently mapped as a pinhole.
    """
    captured = _capture_mapper_kwargs(monkeypatch)
    args = rp.parse_args(
        ["--input", "s.mp4", "--input-projection", "fisheye", "--fisheye-fov", "150", "--output-width", "64"]
    )
    args.output_height = 64
    args.no_ffmpeg_v360 = False

    with pytest.raises(_MapperConstructedError):
        rp.run_equirect_stage(args, [], [])

    assert captured["input_projection"] == "fisheye"
    assert captured["fisheye_fov"] == 150.0


def test_equirect_stage_defaults_the_mapper_to_rectilinear(monkeypatch):
    """Guard the guard: the pass-through above must not hard-code fisheye."""
    captured = _capture_mapper_kwargs(monkeypatch)
    args = rp.parse_args(["--input", "s.mp4", "--output-width", "64"])
    args.output_height = 64
    args.no_ffmpeg_v360 = False

    with pytest.raises(_MapperConstructedError):
        rp.run_equirect_stage(args, [], [])

    assert captured["input_projection"] == "rectilinear"
    assert captured["src_hfov"] == 70.0


def test_fisheye_rejects_explicit_src_hfov(capsys):
    """Non-zero exit + readable error: the equidistant model has no pinhole
    hfov, so silently honouring one of the two would misreport the geometry.
    """
    with pytest.raises(SystemExit) as excinfo:
        rp.parse_args(["--input", "source.mp4", "--input-projection", "fisheye", "--src-hfov", "100"])

    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "--src-hfov" in err
    assert "--fisheye-fov" in err


def test_rectilinear_still_accepts_src_hfov():
    """Guard the guard: the rejection above must be scoped to fisheye."""
    args = rp.parse_args(["--input", "source.mp4", "--src-hfov", "100"])

    assert args.src_hfov == 100.0
    assert args._src_hfov_explicit is True


@pytest.mark.parametrize("bad", ["0", "-5", "400", "nan"])
def test_fisheye_fov_out_of_range_exits_non_zero(bad, capsys):
    with pytest.raises(SystemExit) as excinfo:
        rp.parse_args(["--input", "source.mp4", "--input-projection", "fisheye", "--fisheye-fov", bad])

    assert excinfo.value.code != 0
    assert "--fisheye-fov" in capsys.readouterr().err


def test_non_square_fisheye_input_warns(monkeypatch, caplog):
    monkeypatch.setattr(rp.cv2, "VideoCapture", lambda _: _Capture(1920, 1080))
    args = SimpleNamespace(input="source.mp4", input_projection="fisheye", _src_hfov_explicit=False)

    rp.validate_input_projection(args)

    assert "1920x1080" in caplog.text
    assert "square" in caplog.text


def test_square_fisheye_input_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(rp.cv2, "VideoCapture", lambda _: _Capture(2880, 2880))
    args = SimpleNamespace(input="source.mp4", input_projection="fisheye", _src_hfov_explicit=False)

    rp.validate_input_projection(args)

    assert caplog.text == ""


def test_streaming_path_is_not_warned_about_the_fisheye_knobs(caplog):
    """#243 defence: both knobs are threaded into StreamingPipeline, so the
    drop-detector must not name them as ignored.
    """
    assert "input_projection" in rp._STREAMING_SUPPORTED
    assert "fisheye_fov" in rp._STREAMING_SUPPORTED

    # --tile-size is deliberately NOT in the supported set, so it acts as a
    # positive control: if it is not named, the detector short-circuited and
    # the two assertions below would be vacuous.
    args = rp.parse_args(
        [
            "--input", "s.mp4", "--streaming",
            "--input-projection", "fisheye", "--fisheye-fov", "150",
            "--tile-size", "256",
        ]
    )  # fmt: skip
    rp._warn_streaming_unsupported_args(args)

    assert "--tile-size" in caplog.text, "detector did not run; the assertions below prove nothing"
    assert "--input-projection" not in caplog.text
    assert "--fisheye-fov" not in caplog.text


def test_streaming_pipeline_forwards_the_source_projection_to_its_mapper():
    """The stream must not silently degrade a fisheye source to the pinhole
    model — the #120 / #243 silent-drop class.
    """
    from pipeline.streaming_pipeline import StreamingPipeline

    sig = __import__("inspect").signature(StreamingPipeline.__init__)
    assert sig.parameters["input_projection"].default == "rectilinear"
    assert sig.parameters["fisheye_fov"].default == 180.0


def test_sidecar_generation_records_the_source_projection(tmp_path, monkeypatch):
    captured = {}

    def _fake_write_sidecar(path, *, immersive, generation):
        captured["generation"] = generation

    import pipeline.sidecar as sc

    monkeypatch.setattr(sc, "write_sidecar", _fake_write_sidecar)
    args = SimpleNamespace(
        output_width=2880, output_height=2880, preset=None, input_projection="fisheye", fisheye_fov=150.0
    )

    rp._write_sidecar_from_args(str(tmp_path / "out.mp4"), "vr180", args)

    assert captured["generation"]["input_projection"] == "fisheye"
    assert captured["generation"]["fisheye_fov"] == 150.0


def test_sidecar_omits_fisheye_fov_for_non_fisheye_sources(tmp_path, monkeypatch):
    captured = {}

    import pipeline.sidecar as sc

    monkeypatch.setattr(sc, "write_sidecar", lambda p, *, immersive, generation: captured.update(g=generation))
    args = SimpleNamespace(
        output_width=2880, output_height=2880, preset=None, input_projection="rectilinear", fisheye_fov=180.0
    )

    rp._write_sidecar_from_args(str(tmp_path / "out.mp4"), "vr180", args)

    assert captured["g"]["input_projection"] == "rectilinear"
    assert "fisheye_fov" not in captured["g"]
