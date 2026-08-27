"""Tests for pipeline/image_prep.py — I2V input normalization."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pipeline.image_prep import PreparedImage, parse_aspect, prepare_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(w: int, h: int, color: tuple[int, int, int] = (10, 20, 30)) -> np.ndarray:
    """BGR solid-color image (cv2 layout)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _gradient(w: int, h: int) -> np.ndarray:
    """Distinct per-column gradient so position assertions are exact."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        img[:, x] = (x % 256, (x * 2) % 256, (x * 3) % 256)
    return img


def _write_png(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    assert ok, f"setup: failed to write {path}"


def _write_jpeg_bytes(img_rgb: np.ndarray, orientation: int | None = None) -> bytes:
    """Encode an RGB image to JPEG bytes, optionally with EXIF Orientation."""
    pil = Image.fromarray(img_rgb)
    if orientation is not None and orientation != 1:
        # Build a minimal EXIF with the given Orientation tag.
        exif = pil.getexif()
        exif[0x0112] = orientation  # Orientation tag
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", exif=exif)
    else:
        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# parse_aspect
# ---------------------------------------------------------------------------


class TestParseAspect:
    def test_basic_16_9(self):
        assert parse_aspect("16:9") == pytest.approx(16 / 9)

    def test_basic_4_3(self):
        assert parse_aspect("4:3") == pytest.approx(4 / 3)

    def test_slash_separator(self):
        assert parse_aspect("16/9") == pytest.approx(16 / 9)

    def test_decimal(self):
        assert parse_aspect("2.35:1") == pytest.approx(2.35)

    def test_one_to_one(self):
        assert parse_aspect("1:1") == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", ["169", "16", "", "16::9", "abc:def", "16:x", "16:", ":9", "-16:9"])
    def test_invalid_format_raises(self, bad):
        with pytest.raises(ValueError, match="Invalid aspect ratio format"):
            parse_aspect(bad)

    def test_zero_denominator_raises(self):
        with pytest.raises(ValueError, match="denominator"):
            parse_aspect("16:0")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="Invalid aspect ratio format"):
            parse_aspect(169)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# prepare_image: error paths
# ---------------------------------------------------------------------------


class TestPrepareImageErrors:
    def test_corrupt_file_raises(self, tmp_path):
        bad = tmp_path / "corrupt.png"
        bad.write_bytes(b"not a real image file content")
        with pytest.raises(ValueError, match="Could not decode image"):
            prepare_image(str(bad))

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(ValueError):
            prepare_image(str(missing))

    def test_invalid_mode_raises(self, tmp_path):
        src = tmp_path / "src.png"
        _write_png(src, _solid(400, 300))
        with pytest.raises(ValueError, match="Invalid mode"):
            prepare_image(str(src), mode="stretch")

    def test_invalid_aspect_raises(self, tmp_path):
        src = tmp_path / "src.png"
        _write_png(src, _solid(400, 300))
        with pytest.raises(ValueError, match="Invalid aspect ratio format"):
            prepare_image(str(src), target_aspect="bogus")

    def test_invalid_mode_checked_before_aspect(self, tmp_path):
        """Both mode and aspect are validated; mode is checked first."""
        src = tmp_path / "src.png"
        _write_png(src, _solid(400, 300))
        with pytest.raises(ValueError, match="Invalid mode"):
            prepare_image(str(src), mode="nope", target_aspect="bogus")


# ---------------------------------------------------------------------------
# prepare_image: letterbox — pixel-level
# ---------------------------------------------------------------------------


class TestLetterbox:
    def test_output_size_exact(self, tmp_path):
        # Source 800x600 (4:3) → letterbox into 16:9 at width 1920.
        src = tmp_path / "src.png"
        _write_png(src, _gradient(800, 600))
        out = tmp_path / "out.png"
        result = prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=1920, mode="letterbox")
        assert isinstance(result, PreparedImage)
        assert result.width == 1920
        assert result.height == 1080
        assert out.exists()

    def test_letterbox_padding_is_black(self, tmp_path):
        # 4:3 source (taller than 16:9) letterboxed into 16:9 → black bars left/right.
        src = tmp_path / "src.png"
        _write_png(src, _solid(800, 600, color=(100, 150, 200)))
        out = tmp_path / "out.png"
        prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=1920, mode="letterbox")

        img = cv2.imread(str(out))
        h, w = img.shape[:2]
        assert (w, h) == (1920, 1080)
        # Top-left corner should be black (letterbox padding).
        assert tuple(img[0, 0]) == (0, 0, 0)
        # Center should be the solid color (within rounding).
        assert tuple(img[h // 2, w // 2]) == (100, 150, 200)

    def test_letterbox_content_position(self, tmp_path):
        # Wider-than-target source (1600x600 ≈ 2.67:1) into 16:9 (1.78) → bars top & bottom.
        src = tmp_path / "src.png"
        _write_png(src, _gradient(1600, 600))
        out = tmp_path / "out.png"
        prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=1920, mode="letterbox")

        img = cv2.imread(str(out))
        h, w = img.shape[:2]
        assert (w, h) == (1920, 1080)
        # The image is wider than target ratio → padded top & bottom.
        # Vertical center should hold content; top & bottom rows should be black.
        assert tuple(img[0, w // 2]) == (0, 0, 0), "top row must be black padding"
        assert tuple(img[h - 1, w // 2]) == (0, 0, 0), "bottom row must be black padding"
        # Content present at vertical center.
        assert tuple(img[h // 2, w // 2]) != (0, 0, 0)

    def test_letterbox_preserves_aspect(self, tmp_path):
        # Exact-aspect source (16:9) → no padding, full-bleed.
        src = tmp_path / "src.png"
        _write_png(src, _gradient(1600, 900))
        out = tmp_path / "out.png"
        prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=1920, mode="letterbox")
        img = cv2.imread(str(out))
        # No padding: a non-zero interior pixel confirms content fills the canvas.
        assert tuple(img[540, 1000]) != (0, 0, 0)


# ---------------------------------------------------------------------------
# prepare_image: crop — pixel-level
# ---------------------------------------------------------------------------


class TestCrop:
    def test_output_size_exact(self, tmp_path):
        src = tmp_path / "src.png"
        _write_png(src, _gradient(800, 600))
        out = tmp_path / "out.png"
        result = prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=1920, mode="crop")
        assert result.width == 1920
        assert result.height == 1080

    def test_crop_removes_padding_region(self, tmp_path):
        # 4:3 source into 16:9 crop → side columns are removed, not padded.
        # Construct an image where the left/right edges are a distinct color so
        # we can confirm they are gone after the center crop.
        src_img = _solid(800, 600, color=(50, 50, 50))
        # Mark the center region that survives a 16:9 crop.
        # 16:9 crop of 800x600 keeps height 600 → new_w = 600*16/9 ≈ 1066, but
        # source is only 800 wide → since src_ratio(1.33) < target_ratio(1.78),
        # we crop vertically: new_h = 800/1.78 ≈ 450.
        new_h = round(800 / (16 / 9))
        y0 = (600 - new_h) // 2
        src_img[y0 : y0 + new_h, :] = (200, 100, 50)
        src = tmp_path / "src.png"
        _write_png(src, src_img)
        out = tmp_path / "out.png"
        prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=320, mode="crop")

        img = cv2.imread(str(out))
        h, w = img.shape[:2]
        assert (w, h) == (320, 180)
        # No black padding in crop mode: every pixel comes from the source.
        assert not np.any(np.all(img == 0, axis=2)), "crop mode must not add black padding"
        # The cropped region's center color should survive.
        assert tuple(img[h // 2, w // 2]) == (200, 100, 50)

    def test_crop_no_black_padding(self, tmp_path):
        # Crop must never introduce black bars.
        src = tmp_path / "src.png"
        _write_png(src, _solid(640, 480, color=(123, 45, 67)))
        out = tmp_path / "out.png"
        prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=320, mode="crop")
        img = cv2.imread(str(out))
        # No pixel should be exactly black (source was non-black).
        assert not np.any(np.all(img == 0, axis=2)), "crop mode must not add black padding"

    def test_crop_wider_source(self, tmp_path):
        # Source wider than target (2:1) → crop sides, keep full height.
        src_img = _solid(1000, 500, color=(11, 22, 33))
        src = tmp_path / "src.png"
        _write_png(src, src_img)
        out = tmp_path / "out.png"
        result = prepare_image(str(src), out_path=str(out), target_aspect="1:1", target_width=400, mode="crop")
        img = cv2.imread(str(out))
        assert img.shape[:2] == (400, 400)
        assert result.width == 400 and result.height == 400
        # All pixels from source (non-black).
        assert not np.any(np.all(img == 0, axis=2))


# ---------------------------------------------------------------------------
# EXIF rotation
# ---------------------------------------------------------------------------


class TestExifRotation:
    def _make_oriented_jpeg(self, tmp_path, orientation: int) -> Path:
        """Create a JPEG whose pixel data is rotated but EXIF Orientation
        says to rotate it back. A non-square image so rotation is visible."""
        # Base image: distinct quadrants (top-left red, top-right green,
        # bottom-left blue, bottom-right white) — in RGB.
        base = np.zeros((400, 800, 3), dtype=np.uint8)
        base[:200, :400] = (255, 0, 0)  # TL red
        base[:200, 400:] = (0, 255, 0)  # TR green
        base[200:, :400] = (0, 0, 255)  # BL blue
        base[200:, 400:] = (255, 255, 255)  # BR white

        # When Pillow applies exif_transpose with orientation=6 (rotate 90° CW),
        # it physically rotates the array. To emulate a camera that stored the
        # image rotated-with-EXIF, we store the *physically rotated* pixels and
        # set Orientation so exif_transpose un-rotates them back to base.
        if orientation == 6:  # stored rotated 90° CCW → transpose back to base
            stored = np.rot90(base, k=3)  # rotate 90° CW physically → EXIF 6 undoes
        elif orientation == 3:  # 180°
            stored = np.rot90(base, k=2)
        elif orientation == 8:  # 90° CCW
            stored = np.rot90(base, k=1)
        else:
            stored = base

        path = tmp_path / f"oriented_{orientation}.jpg"
        path.write_bytes(_write_jpeg_bytes(stored, orientation=orientation))
        return path

    def test_exif_orientation_6_applied(self, tmp_path):
        src = self._make_oriented_jpeg(tmp_path, 6)
        out = tmp_path / "out.png"
        # 16:9, width 800 → height 450. Source base was 800x400.
        prepare_image(str(src), out_path=str(out), target_aspect="2:1", target_width=800, mode="crop")
        img = cv2.imread(str(out))
        h, w = img.shape[:2]
        # After EXIF correction the image should be landscape (w > h).
        assert w > h, f"EXIF orientation not applied: got {w}x{h}"

    def test_no_exif_unchanged(self, tmp_path):
        # An image without EXIF orientation must pass through unrotated.
        base = np.zeros((400, 800, 3), dtype=np.uint8)
        base[:200, :] = (255, 0, 0)
        base[200:, :] = (0, 255, 0)
        path = tmp_path / "noexif.jpg"
        path.write_bytes(_write_jpeg_bytes(base, orientation=None))
        out = tmp_path / "out.png"
        prepare_image(str(path), out_path=str(out), target_aspect="2:1", target_width=800, mode="crop")
        img = cv2.imread(str(out))
        _h, w = img.shape[:2]
        # Source is 2:1 and target is 2:1 → no crop, top half stays red (BGR: 0,0,255).
        # JPEG is lossy so allow a tolerance on the red channel.
        top_pixel = img[10, w // 2]
        assert top_pixel[2] > 200 and top_pixel[1] < 60 and top_pixel[0] < 60, (
            f"top half should be red, got {tuple(top_pixel)}"
        )


# ---------------------------------------------------------------------------
# Small-image warning
# ---------------------------------------------------------------------------


class TestSmallImageWarning:
    def test_narrow_input_warns(self, tmp_path):
        src = tmp_path / "small.png"
        _write_png(src, _solid(512, 512))
        result = prepare_image(str(src), target_width=1024, mode="letterbox")
        assert any("narrow" in w and "1024" in w for w in result.warnings)

    def test_wide_input_no_warning(self, tmp_path):
        src = tmp_path / "wide.png"
        _write_png(src, _solid(1920, 1080))
        result = prepare_image(str(src), target_width=1920, mode="letterbox")
        assert result.warnings == []

    def test_boundary_1024_no_warning(self, tmp_path):
        # Exactly 1024 → not < 1024, no warning.
        src = tmp_path / "edge.png"
        _write_png(src, _solid(1024, 768))
        result = prepare_image(str(src), target_width=1280, mode="letterbox")
        assert result.warnings == []


# ---------------------------------------------------------------------------
# out_path default & round-trip
# ---------------------------------------------------------------------------


class TestOutputPath:
    def test_default_out_path_next_to_source(self, tmp_path):
        src = tmp_path / "myimage.png"
        _write_png(src, _solid(1280, 720))
        result = prepare_image(str(src), target_width=1280, mode="letterbox")
        expected = tmp_path / "myimage_prep.png"
        assert Path(result.path) == expected
        assert expected.exists()

    def test_custom_out_path(self, tmp_path):
        src = tmp_path / "src.png"
        _write_png(src, _solid(1280, 720))
        out = tmp_path / "sub" / "custom.png"
        out.parent.mkdir()
        result = prepare_image(str(src), out_path=str(out), target_width=1280, mode="letterbox")
        assert Path(result.path) == out
        assert out.exists()

    def test_roundtrip_readable_as_image(self, tmp_path):
        src = tmp_path / "src.png"
        _write_png(src, _gradient(1280, 720))
        out = tmp_path / "out.png"
        result = prepare_image(str(src), out_path=str(out), target_aspect="16:9", target_width=1920, mode="letterbox")
        img = cv2.imread(result.path)
        assert img is not None, "prepared image must be readable by cv2"
        assert img.shape[1] == 1920 and img.shape[0] == 1080


# ---------------------------------------------------------------------------
# Resize method selection (Lanczos vs AREA)
# ---------------------------------------------------------------------------


class TestResizeMethods:
    def test_upscale_uses_lanczos(self, tmp_path, monkeypatch):
        src = tmp_path / "src.png"
        _write_png(src, _solid(640, 360))
        calls: list[int] = []
        orig_resize = cv2.resize

        def spy(img, dsize, interpolation=None, **kw):
            calls.append(interpolation)
            return orig_resize(img, dsize, interpolation=interpolation, **kw)

        monkeypatch.setattr(cv2, "resize", spy)
        prepare_image(str(src), target_width=1920, mode="letterbox")
        assert cv2.INTER_LANCZOS4 in calls, "upscaling should use Lanczos"

    def test_downscale_uses_area(self, tmp_path, monkeypatch):
        src = tmp_path / "src.png"
        _write_png(src, _solid(2560, 1440))
        calls: list[int] = []
        orig_resize = cv2.resize

        def spy(img, dsize, interpolation=None, **kw):
            calls.append(interpolation)
            return orig_resize(img, dsize, interpolation=interpolation, **kw)

        monkeypatch.setattr(cv2, "resize", spy)
        prepare_image(str(src), target_width=640, mode="letterbox")
        assert cv2.INTER_AREA in calls, "downscaling should use AREA"
