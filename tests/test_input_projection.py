"""Tests for the --input-projection equirect source path (#239)."""

from types import SimpleNamespace

import numpy as np
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
