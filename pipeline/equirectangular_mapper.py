"""
Stage 3 — Equirectangular Projection
=====================================

Map planar stereo views onto a 180° hemisphere using equirectangular
projection. The output is a 7680×1920 SBS frame (3840×1920 per eye).

Two strategies:
1. **ffmpeg v360 filter** (recommended) — Uses ffmpeg's built-in
   v360 filter for fast, correct perspective→equirectangular mapping.
2. **OpenCV remap** (fallback) — Pure NumPy/OpenCV implementation.

Batched mode (default):
  Instead of calling ffmpeg 2×N times (once per eye per frame), the
  ``map_sequence()`` method writes all frames as a temporary image
  sequence and runs ffmpeg **once per eye** on the whole video.
  This yields ~10× speedup for long clips.

Usage:
    from pipeline.equirectangular_mapper import EquirectangularMapper
    mapper = EquirectangularMapper()
    sbs_frame = mapper.map_stereo_pair(left_frame, right_frame)
    sbs_frames = mapper.map_sequence(left_frames, right_frames, temp_dir)
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


class EquirectangularMapper:
    """Map planar stereo views to VR180 equirectangular format.

    Output: 7680×1920 SBS frame (2 × 3840×1920 hemispheres).

    Regions outside the source FOV are **pure black RGB (0,0,0)** on *both*
    paths (issue #255).  This is a terminal state, not an intermediate one:
    the output of this class is encoded to H.264/HEVC ``yuv420p``, which
    carries no alpha, so anything that relies on alpha to hide the hole is
    discarded at encode time.

    - **ffmpeg v360 path** (default, ``use_ffmpeg=True``): ``alpha_mask=1``
      marks the uncovered region with alpha=0, and the *same* filter chain
      immediately composites that RGBA frame over a black background
      (``split`` → ``lutrgb`` black → ``overlay``).  What comes back is
      genuinely black outside the FOV, not v360's default edge-pixel smear.
      Compositing happens inside ffmpeg — no per-frame NumPy pass.
    - **OpenCV remap path** (fallback, ``use_ffmpeg=False``): out-of-FOV
      pixels are remapped with a constant black border and then forced to
      RGB (0,0,0).

    ``map_single`` returns 3-channel RGB by default; ``with_alpha=True``
    additionally keeps the alpha plane (0 outside the FOV) for callers that
    need a real mask for feathering/compositing.  The RGB planes are identical
    either way.
    """

    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1920,
        src_hfov: float = 90.0,
        use_ffmpeg: bool = True,
    ):
        """Configure the equirectangular mapper.

        Args:
            output_width: Per-eye equirectangular width (px).
                Default 1920 → square 1:1 per eye for comfortable VR180.
                3840 gives sharper full-resolution output at higher render cost.
            output_height: Per-eye equirectangular height (px).
                Default 1920 (square 1:1). Matches output_width for square per-eye.
            src_hfov: Source camera horizontal field of view (degrees).
                Default 90° — good tradeoff for most AI-generated and action-cam
                footage. Higher (e.g. 120°) fills more of the 180° dome but
                introduces more peripheral stretch. Lower (e.g. 70°) gives a
                "binoculars" feel with less stretch but worse immersion.
            use_ffmpeg: Prefer ffmpeg v360 filter when available.
        """
        self.output_width = output_width
        self.output_height = output_height
        self.src_hfov = src_hfov
        self.use_ffmpeg = use_ffmpeg
        self._mesh: tuple[np.ndarray, np.ndarray] | None = None

    def map_single(self, frame: np.ndarray, with_alpha: bool = False) -> np.ndarray:
        """Map a single planar frame to equirectangular VR180.

        Pixels outside the source FOV are pure black RGB (0,0,0) regardless of
        ``with_alpha`` — see the class docstring and issue #255.

        Args:
            frame: Input planar image (H, W, 3), uint8.
            with_alpha: When True, return an RGBA frame whose alpha plane is 0
                for pixels that fall outside the source FOV. Default False →
                the 3-channel RGB contract used by :meth:`map_stereo_pair` and
                downstream consumers. The RGB planes are identical either way.

        Returns:
            Equirectangular frame (output_height, output_width, 3 or 4), uint8.
        """
        if self.use_ffmpeg and self._ffmpeg_available():
            return self._map_via_ffmpeg(frame, with_alpha=with_alpha)
        else:
            return self._map_via_opencv(frame, with_alpha=with_alpha)

    def _ffmpeg_available(self) -> bool:
        """Check if ffmpeg with v360 filter is available."""
        import shutil

        if not shutil.which("ffmpeg"):
            return False
        try:
            result = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5)
            return "v360" in result.stdout
        except Exception:
            return False

    def _calc_vertical_fov(self, src_width: int, src_height: float) -> float:
        """Calculate vertical FOV from horizontal FOV and aspect ratio.

        For a pinhole camera: vfov = 2 * atan(tan(hfov/2) * height/width)
        """
        import math

        hfov_rad = math.radians(self.src_hfov)
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * src_height / src_width)
        return math.degrees(vfov_rad)

    #: Composite an ``alpha_mask=1`` v360 frame onto black, inside ffmpeg.
    #:
    #: ``alpha_mask=1`` alone only zeroes the *alpha* plane — the RGB planes
    #: still hold v360's edge-clamped smear, and alpha is thrown away the
    #: moment the frame is encoded to ``yuv420p`` (issue #255).  So the hole
    #: has to become black RGB before the frame ever leaves ffmpeg.
    #:
    #: The black background is derived from the v360 output itself via
    #: ``split`` + ``lutrgb`` rather than a ``color=black`` lavfi source: the
    #: two branches then share timestamps and frame count by construction, so
    #: the chain behaves identically for a single PNG, an image sequence and a
    #: full video (a ``color`` source has its own frame rate and would need
    #: per-call rate matching).
    #:
    #: ``lutrgb`` leaves the alpha plane untouched, so straight-alpha
    #: ``overlay`` yields ``out_a = fg_a`` — the mask survives for
    #: ``with_alpha=True`` callers while the RGB is already composited.
    #: ``overlay`` (unlike ``premultiply=inplace=1``, which rounds
    #: ``rgb*alpha/255`` down and darkens every in-FOV pixel by 1 LSB) leaves
    #: covered pixels bit-exact.
    _BLACK_COMPOSITE = (
        "split[_fg][_bgsrc];[_bgsrc]lutrgb=r=0:g=0:b=0[_bg];[_bg][_fg]overlay=format=rgb:eof_action=endall"
    )

    def _v360_filter(self, src_width: int, src_height: int, with_alpha: bool = False) -> str:
        """Build the perspective → half-equirectangular filter chain.

        The chain is ``v360=...:alpha_mask=1`` followed by
        :attr:`_BLACK_COMPOSITE`, so out-of-FOV pixels come back as real black
        RGB rather than v360's edge smear (issue #255).

        Args:
            src_width: Source frame width (px), for the vertical-FOV solve.
            src_height: Source frame height (px).
            with_alpha: Terminate the chain in ``rgba`` (keeping the alpha
                mask, 0 outside the FOV) instead of ``rgb24``. The RGB planes
                are identical either way.
        """
        src_vfov = self._calc_vertical_fov(src_width, src_height)
        v360 = (
            f"v360=input=flat:output=hequirect:"
            f"ih_fov={self.src_hfov}:iv_fov={src_vfov:.2f}:"
            f"h_fov=180:v_fov=180:"
            f"w={self.output_width}:h={self.output_height}:"
            f"alpha_mask=1"
        )
        pix_fmt = "rgba" if with_alpha else "rgb24"
        return f"{v360},{self._BLACK_COMPOSITE},format={pix_fmt}"

    def _map_via_ffmpeg(self, frame: np.ndarray, with_alpha: bool = False) -> np.ndarray:
        """Use ffmpeg v360 filter for equirectangular mapping.

        Maps a flat perspective image (with src_hfov FOV) onto a
        180° hemispherical equirectangular projection.

        The v360 output is composited onto black **inside the same ffmpeg
        invocation** (see :attr:`_BLACK_COMPOSITE` and issue #255), so the RGB
        that comes back is genuinely (0,0,0) outside the source FOV — no
        per-frame NumPy post-pass, and nothing that depends on alpha surviving
        the ``yuv420p`` encode downstream.
        """
        import tempfile

        H, W = frame.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as in_f:
            in_path = in_f.name
            cv2.imwrite(in_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out_path = in_path.replace(".png", "_eq.png")
        try:
            vfilter = self._v360_filter(W, H, with_alpha=with_alpha)
            cmd = ["ffmpeg", "-y", "-i", in_path, "-vf", vfilter, "-frames:v", "1", out_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)

            out_img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
            if out_img is None:
                raise RuntimeError("ffmpeg v360 failed to produce output")
            if with_alpha:
                if out_img.ndim != 3 or out_img.shape[2] != 4:
                    raise RuntimeError(f"ffmpeg v360 produced {out_img.shape} where RGBA was requested")
                return cv2.cvtColor(out_img, cv2.COLOR_BGRA2RGBA)
            # The chain terminates in ``format=rgb24``; drop any stray alpha
            # defensively so the 3-channel contract holds unconditionally.
            return cv2.cvtColor(out_img[:, :, :3], cv2.COLOR_BGR2RGB)
        finally:
            try:
                os.unlink(in_path)
                if os.path.exists(out_path):
                    os.unlink(out_path)
            except OSError:
                pass

    def _map_via_opencv(self, frame: np.ndarray, with_alpha: bool = False) -> np.ndarray:
        """Pure OpenCV equirectangular mapping (fallback).

        Pixels outside the source camera's FOV are filled with pure black (RGB
        0,0,0) — remapped with a constant black border and masked explicitly.
        With ``with_alpha=True`` those pixels also carry alpha=0, mirroring the
        ffmpeg path's real alpha hole; RGB is unchanged either way.
        """
        import cv2

        H_src, W_src = frame.shape[:2]

        if self._mesh is None:
            self._build_mesh(W_src, H_src)

        sx, sy = self._mesh

        # Create mask for valid pixels (those within source bounds)
        valid_mask = sx >= 0

        # Replace invalid coords with 0 for remap (will be masked later)
        sx_safe = np.where(valid_mask, sx, 0.0).astype(np.float32)
        sy_safe = np.where(valid_mask, sy, 0.0).astype(np.float32)

        equirect = cv2.remap(
            frame, sx_safe, sy_safe, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # Apply black fill for out-of-FOV regions
        if not np.all(valid_mask):
            equirect[~valid_mask] = [0, 0, 0]

        if not with_alpha:
            return equirect

        rgba = np.dstack([equirect, np.where(valid_mask, 255, 0).astype(np.uint8)])
        return rgba

    def _build_mesh(self, src_width: int, src_height: int):
        """Pre-compute equirectangular→planar mapping mesh.

        For each output pixel (u, v) in the equirect frame:
          1. Compute spherical direction (theta, phi)
          2. Project to the flat source camera's sensor plane
          3. Sample at (sx, sy)

        Pixels outside the source camera's FOV are marked as -1
        and filled with black instead of being stretched.
        """
        import math

        W_out, H_out = self.output_width, self.output_height

        # Output pixel grid
        u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))
        u = u.astype(np.float32)
        v = v.astype(np.float32)

        # Spherical coordinates for a 180° hemisphere
        # theta: -90° to +90° (horizontal), phi: 0° (top) to 180° (bottom)
        theta = (u / W_out - 0.5) * np.pi  # [-π/2, π/2]
        phi = (v / H_out) * np.pi  # [0, π]

        # Ray direction in 3D
        ray_x = np.sin(theta) * np.sin(phi)
        ray_y = np.cos(phi)
        ray_z = np.cos(theta) * np.sin(phi)

        # Project onto source camera plane (pinhole model)
        hfov_rad = math.radians(self.src_hfov)
        fx = src_width / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx  # square pixels assumed

        cx, cy = src_width / 2.0, src_height / 2.0

        # Mask: only project rays that are in front of the camera (ray_z > 0)
        valid = ray_z > 0.01

        # ray_y is positive-up; image y is positive-down → negate for correct mapping
        sx = np.where(valid, fx * ray_x / np.maximum(ray_z, 1e-6) + cx, -1.0)
        sy = np.where(valid, -fy * ray_y / np.maximum(ray_z, 1e-6) + cy, -1.0)

        # Check if projected point is within source image bounds
        in_bounds = valid & (sx >= 0) & (sx < src_width) & (sy >= 0) & (sy < src_height)

        # Mark out-of-bounds pixels for black fill
        sx = np.where(in_bounds, sx, -1.0)
        sy = np.where(in_bounds, sy, -1.0)

        self._mesh = (sx.astype(np.float32), sy.astype(np.float32))

    def map_stereo_pair(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """Map left+right views into a SBS equirectangular frame.

        Each view is independently mapped to equirectangular,
        then concatenated side-by-side → 7680×1920.

        Args:
            left_frame: Left eye planar image (H, W, 3), uint8
            right_frame: Right eye planar image (H, W, 3), uint8

        Returns:
            SBS equirect frame (H_out, W_out*2, 3), uint8
        """
        left_eq = self.map_single(left_frame)
        right_eq = self.map_single(right_frame)
        return np.concatenate([left_eq, right_eq], axis=1)

    # ------------------------------------------------------------------
    # Batched processing — ~10× faster than per-frame ffmpeg calls
    # ------------------------------------------------------------------

    def _write_image_sequence(
        self,
        frames: list[np.ndarray],
        prefix: str,
        output_dir: str,
    ) -> tuple[int, int]:
        """Write frames as a PNG image sequence and return (height, width) of first frame."""
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"{prefix}_{i:06d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        H, W = frames[0].shape[:2]
        return H, W

    def _read_image_sequence(
        self,
        prefix: str,
        num_frames: int,
        input_dir: str,
    ) -> list[np.ndarray]:
        """Read back an image sequence as RGB ndarrays."""
        result = []
        for i in range(num_frames):
            path = os.path.join(input_dir, f"{prefix}_{i:06d}.png")
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Missing frame: {path}")
            result.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return result

    def _run_ffmpeg_v360_on_dir(
        self,
        pattern: str,
        output_dir: str,
        out_prefix: str,
        w: int,
        h: int,
        num_frames: int,
    ):
        """Run ffmpeg v360 **once** on an image sequence directory.

        Writes output frames as ``{out_prefix}_000000.png`` etc.
        """
        vfilter = self._v360_filter(w, h)

        # Use %06d pattern for glob input
        in_pattern = os.path.join(output_dir, pattern).replace("\\", "/")
        out_pattern = os.path.join(output_dir, f"{out_prefix}_%06d.png").replace("\\", "/")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            in_pattern,
            "-vf",
            vfilter,
            "-frames:v",
            str(num_frames),
            "-start_number",
            "0",
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

    def map_sequence(
        self,
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        temp_dir: str,
    ) -> list[np.ndarray]:
        """Map a sequence of left/right frames to SBS equirectangular in batch.

        Instead of calling ffmpeg 2×N times, writes PNG sequences to
        ``temp_dir``, runs ffmpeg v360 **once per eye** on the full
        sequence, then reads back the results.  ~10× faster for long clips.

        Falls back to per-frame OpenCV mapping if ffmpeg v360 is unavailable.

        Args:
            left_frames: List of left eye frames (H, W, 3), uint8
            right_frames: List of right eye frames (H, W, 3), uint8
            temp_dir: Writable directory for intermediate PNG sequences.

        Returns:
            List of SBS equirect frames (H_out, W_out*2, 3), uint8
        """
        if not left_frames or not right_frames:
            raise ValueError("Empty frame lists")

        if self.use_ffmpeg and self._ffmpeg_available():
            return self._map_sequence_via_ffmpeg(left_frames, right_frames, temp_dir)
        else:
            # Fallback: call map_single per frame via existing per-frame path
            return [self.map_stereo_pair(left, right) for left, right in zip(left_frames, right_frames, strict=False)]

    def _map_sequence_via_ffmpeg(
        self,
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
        temp_dir: str,
    ) -> list[np.ndarray]:
        """Batch equirect via single ffmpeg call per eye."""
        tmp = Path(temp_dir) / "_equirect_batch"
        tmp.mkdir(parents=True, exist_ok=True)
        num = len(left_frames)

        # 1. Write input PNG sequences
        lw, lh = self._write_image_sequence(left_frames, "L_in", str(tmp))
        rw, rh = self._write_image_sequence(right_frames, "R_in", str(tmp))

        # 2. Run ffmpeg v360 once per eye on the whole sequence
        self._run_ffmpeg_v360_on_dir("L_in_%06d.png", str(tmp), "L_out", lw, lh, num)
        self._run_ffmpeg_v360_on_dir("R_in_%06d.png", str(tmp), "R_out", rw, rh, num)

        # 3. Read back equirect results
        left_eq = self._read_image_sequence("L_out", num, str(tmp))
        right_eq = self._read_image_sequence("R_out", num, str(tmp))

        # 4. Build SBS pairs
        sbs_frames = [np.concatenate([left, right], axis=1) for left, right in zip(left_eq, right_eq, strict=False)]

        # 5. Cleanup temp images (keep the dir itself for cache)
        import contextlib

        for fname in os.listdir(str(tmp)):
            fp = os.path.join(str(tmp), fname)
            with contextlib.suppress(OSError):
                os.unlink(fp)

        return sbs_frames

    def map_video(
        self,
        left_video: str,
        right_video: str,
        temp_dir: str,
        output_path: str,
        fps: int = 30,
    ) -> str:
        """Map an entire left/right eye video pair to an equirect SBS video.

        Uses a **single** ffmpeg v360 pass per eye on the whole video
        (if ffmpeg v360 is available), then concats left+right eq
        frames into SBS video.

        Falls back to per-frame OpenCV mapping if v360 filter
        is unavailable.

        Args:
            left_video: Path to left eye video file.
            right_video: Path to right eye video file.
            temp_dir: Temporary directory for frame extraction.
            output_path: Path for the output SBS equirectangular video.
            fps: Output framerate.

        Returns:
            ``output_path`` on success.
        """
        if self.use_ffmpeg and self._ffmpeg_available():
            return self._map_video_via_ffmpeg(left_video, right_video, temp_dir, output_path, fps)

        # Fallback: extract frames, map per-frame, re-encode
        extract_dir = Path(temp_dir) / "_equirect_vid_frames"
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract left frames
        left_pat = str(extract_dir / "L_%06d.png").replace("\\", "/")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                left_video,
                "-vf",
                f"fps={fps}",
                "-start_number",
                "0",
                left_pat,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

        # Extract right frames
        right_pat = str(extract_dir / "R_%06d.png").replace("\\", "/")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                right_video,
                "-vf",
                f"fps={fps}",
                "-start_number",
                "0",
                right_pat,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

        # Count frames
        left_files = sorted(extract_dir.glob("L_*.png"))
        right_files = sorted(extract_dir.glob("R_*.png"))
        num = min(len(left_files), len(right_files))

        left_frames: list[np.ndarray] = []
        right_frames: list[np.ndarray] = []
        for i in range(num):
            lf = cv2.imread(str(left_files[i]))
            rf = cv2.imread(str(right_files[i]))
            if lf is None or rf is None:
                raise RuntimeError(f"Missing extracted frame at index {i}")
            left_frames.append(cv2.cvtColor(lf, cv2.COLOR_BGR2RGB))
            right_frames.append(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB))

        sbs_frames = self.map_sequence(left_frames, right_frames, temp_dir)

        # Re-encode to output video
        out_pat = str(extract_dir / "SBS_%06d.png").replace("\\", "/")
        for i, sbs in enumerate(sbs_frames):
            cv2.imwrite(out_pat.replace("%06d", f"{i:06d}"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                out_pat,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )

        # Cleanup frame dir
        import shutil

        shutil.rmtree(str(extract_dir), ignore_errors=True)

        return output_path

    def _map_video_via_ffmpeg(
        self,
        left_video: str,
        right_video: str,
        temp_dir: str,
        output_path: str,
        fps: int = 30,
    ) -> str:
        """Batch equirectangular mapping of a whole video pair.

        Uses a single ffmpeg v360 filter per eye, avoiding per-frame
        spawning overhead entirely.
        """
        # Determine source dimensions from first video
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                left_video,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        parts = probe.stdout.strip().split(",")
        w, h = int(parts[0]), int(parts[1])

        vfilter = self._v360_filter(w, h)

        def _encode_eye(input_video: str, tag: str) -> str:
            outpath = os.path.join(temp_dir, f"_eq_eye_{tag}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                input_video,
                "-vf",
                vfilter,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                outpath,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            return outpath

        left_eq = _encode_eye(left_video, "L")
        right_eq = _encode_eye(right_video, "R")

        # Concatenate side-by-side via hstack filter
        cmd_sbs = [
            "ffmpeg",
            "-y",
            "-i",
            left_eq,
            "-i",
            right_eq,
            "-filter_complex",
            "hstack=inputs=2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(cmd_sbs, check=True, capture_output=True, timeout=600)

        # Cleanup intermediate files
        import contextlib

        for f in [left_eq, right_eq]:
            with contextlib.suppress(OSError):
                os.unlink(f)

        return output_path
