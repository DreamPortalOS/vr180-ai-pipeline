"""
Stage 0 / Stage 3.5 — Pixel Upscaling
Super-resolution using Real-ESRGAN for enhanced visual quality.
Supports 2× and 4× upscaling with MPS/CUDA/CPU auto-detection.
"""

from typing import ClassVar

import cv2
import numpy as np


class PixelUpscaler:
    """Real-ESRGAN based image/video super-resolution.

    Uses Real-ESRGAN (x2plus or x4plus) models for perceptually
    faithful upscaling. Works on individual frames (NumPy arrays).

    Recommended usage:
        - Upscale AFTER equirectangular mapping (Stage 3.5)
        - Or upscale BEFORE depth estimation (Stage 0) for better depth

    Args:
        scale: Upscale factor (2 or 4)
        model_name: 'RealESRGAN_x2plus' or 'RealESRGAN_x4plus'
        device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        half_precision: Use fp16 for faster inference on CUDA
    """

    MODELS: ClassVar[dict] = {
        "RealESRGAN_x2plus": {
            "scale": 2,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        },
        "RealESRGAN_x4plus": {
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        },
        "realesr-animevideov3": {
            "scale": 4,
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        },
    }

    def __init__(
        self,
        scale: int = 2,
        model_name: str | None = None,
        device: str | None = None,
        half_precision: bool = False,
    ):
        self.scale = scale
        self.model_name = model_name or f"RealESRGAN_x{scale}plus"
        self.device = self._resolve_device(device)
        self.half = half_precision and self.device == "cuda"
        self._upsampler = None
        self._use_opencv_fallback = False

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device.lower()
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        if self._upsampler is not None:
            return

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            # Fall back to OpenCV-based upscaling
            self._use_opencv_fallback = True
            self._upsampler = "opencv_fallback"
            print(f"[Upscale] Real-ESRGAN not installed, using OpenCV bicubic fallback (scale={self.scale}×)")
            return

        model_info = self.MODELS.get(self.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(self.MODELS.keys())}")

        # Build network architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=model_info["scale"],
        )

        # Determine model path
        model_path = self._get_model_path(self.model_name)

        # For MPS, use CPU as Real-ESRGAN doesn't natively support MPS
        # PyTorch MPS works but with some operations falling back to CPU
        outscale = self.scale
        upsampler_device = self.device
        if self.device == "mps":
            upsampler_device = "cpu"  # Real-ESRGAN's custom CUDA ops don't support MPS

        self._upsampler = RealESRGANer(
            scale=model_info["scale"],
            model_path=model_path,
            model=model,
            tile=400,  # Tile size to avoid OOM
            tile_pad=10,
            pre_pad=0,
            half=self.half,
            device=upsampler_device,
        )
        self._outscale = outscale
        print(f"[Upscale] Loaded {self.model_name} on {upsampler_device} (scale={self.scale}×)")

    @staticmethod
    def _get_model_path(model_name: str) -> str:
        """Get or download model weights."""
        import os

        weights_dir = os.path.join(os.path.expanduser("~"), ".cache", "realesrgan")
        os.makedirs(weights_dir, exist_ok=True)
        model_path = os.path.join(weights_dir, f"{model_name}.pth")

        if not os.path.exists(model_path):
            url = PixelUpscaler.MODELS[model_name]["url"]
            print(f"[Upscale] Downloading {model_name} weights...")
            import torch

            torch.hub.download_url_to_file(url, model_path)
            print(f"[Upscale] Downloaded to {model_path}")

        return model_path

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single frame (BGR numpy array).

        Args:
            frame: Input image as numpy array (H, W, 3) in BGR uint8

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, 3) in BGR uint8
        """
        self._load_model()

        if getattr(self, "_use_opencv_fallback", False):
            h, w = frame.shape[:2]
            return cv2.resize(
                frame,
                (w * self.scale, h * self.scale),
                interpolation=cv2.INTER_CUBIC,
            )

        # Real-ESRGAN expects BGR uint8
        output, _ = self._upsampler.enhance(frame, outscale=self._outscale)
        return output

    def upscale_tiled(
        self,
        frame: np.ndarray,
        tile_size: int = 512,
        tile_pad: int = 10,
        blend_margin: int = 16,
        blend_mode: str = "gaussian",
        progress_callback=None,
    ) -> np.ndarray:
        """Upscale a large frame using tile-by-tile processing with seamless blending.

        Splits the input frame into tiles of tile_size×tile_size,
        upscales each tile independently via Real-ESRGAN, then stitches
        them back together using Gaussian/Linear feathered blending across
        overlapping margin regions to eliminate visible seam artifacts.

        Based on PRD Section 7.4 — Tiled Upscaling for 8K.

        Args:
            frame: Input image as numpy array (H, W, 3) in BGR uint8
            tile_size: Size of each processing tile (default 512)
            tile_pad: Padding around each tile for context (default 10)
            blend_margin: Width of the feathered blending ramp in output pixels.
                          Tiles overlap by this amount and alpha-blend across it.
                          Set to 0 to disable blending (fast but may show seams).
            blend_mode: 'gaussian' for smooth Gaussian falloff, 'linear' for
                        simple linear ramp. Gaussian produces smoother results.
            progress_callback: Optional callback(current_tile, total_tiles)

        Returns:
            Upscaled image as numpy array (H*scale, W*scale, 3) in BGR uint8
        """
        self._load_model()
        import torch

        h, w = frame.shape[:2]
        scale = self._outscale
        out_h, out_w = h * scale, w * scale

        # We use a weighted accumulator and a weight map for seamless blending.
        # Each tile contributes its pixels weighted by a 2D blending mask.
        # Final output = sum(tile_pixels * weight) / sum(weight).
        output_accum = np.zeros((out_h, out_w, 3), dtype=np.float64)
        weight_map = np.zeros((out_h, out_w), dtype=np.float64)

        # Calculate tile grid with overlap for blending
        stride = tile_size - blend_margin  # overlap by blend_margin in source space
        if blend_margin <= 0 or stride <= 0:
            stride = tile_size  # no overlap

        # Ensure we cover the entire image
        tile_positions = []
        ty = 0
        while ty < h:
            tx = 0
            while tx < w:
                tile_positions.append((tx, ty))
                tx += stride
            ty += stride

        # Also ensure we include the right/bottom edges exactly
        # (the while loops above may overshoot, which is fine — we clamp in extraction)

        total_tiles = len(tile_positions)
        tile_count = 0

        for tx, ty in tile_positions:
            # Source tile boundaries (with padding, clamped to image bounds)
            x0 = tx
            y0 = ty
            x1 = min(tx + tile_size, w)
            y1 = min(ty + tile_size, h)

            # Actual tile dimensions (may be smaller at edges)
            tile_w = x1 - x0
            tile_h = y1 - y0

            # Add context padding
            pad_x0 = max(0, x0 - tile_pad)
            pad_y0 = max(0, y0 - tile_pad)
            pad_x1 = min(w, x1 + tile_pad)
            pad_y1 = min(h, y1 + tile_pad)

            # Extract padded tile
            tile = frame[pad_y0:pad_y1, pad_x0:pad_x1].copy()

            # Upscale tile
            try:
                upscaled_tile, _ = self._upsampler.enhance(tile, outscale=scale)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    uh, uw = tile.shape[:2]
                    upscaled_tile = cv2.resize(
                        tile,
                        (uw * scale, uh * scale),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    print(f"[Upscale] OOM on tile ({tx},{ty}), used lanczos fallback")
                else:
                    raise

            # Remove padding from upscaled tile to get the core region
            inner_x0 = (x0 - pad_x0) * scale
            inner_y0 = (y0 - pad_y0) * scale
            inner_x1 = inner_x0 + tile_w * scale
            inner_y1 = inner_y0 + tile_h * scale
            cropped = upscaled_tile[inner_y0:inner_y1, inner_x0:inner_x1]

            crop_h, crop_w = cropped.shape[:2]
            out_x0 = x0 * scale
            out_y0 = y0 * scale

            # Generate 2D blending weight for this tile
            weight = self._compute_tile_weight(crop_h, crop_w, blend_margin * scale, blend_mode)

            # Accumulate weighted pixels
            output_accum[out_y0 : out_y0 + crop_h, out_x0 : out_x0 + crop_w] += (
                cropped.astype(np.float64) * weight[:, :, np.newaxis]
            )
            weight_map[out_y0 : out_y0 + crop_h, out_x0 : out_x0 + crop_w] += weight

            tile_count += 1  # noqa: SIM113 — explicit counter reads clearer than enumerate here
            if progress_callback:
                progress_callback(tile_count, total_tiles)

        # Normalize by accumulated weights (avoid division by zero)
        weight_map = np.maximum(weight_map, 1e-8)
        output = (output_accum / weight_map[:, :, np.newaxis]).astype(np.uint8)

        return output

    @staticmethod
    def _compute_tile_weight(height: int, width: int, margin_px: int, mode: str = "gaussian") -> np.ndarray:
        """Compute a 2D blending weight map for a single tile.

        The weight is 1.0 in the center and ramps down to near 0 at the edges
        using either a Gaussian or linear falloff across the margin region.

        Args:
            height: Tile height in output pixels
            width: Tile width in output pixels
            margin_px: Width of the feathering ramp in pixels
            mode: 'gaussian' or 'linear'

        Returns:
            2D weight array (H, W) with values in [0, 1]
        """
        if margin_px <= 0:
            return np.ones((height, width), dtype=np.float64)

        # Clamp margin to half the tile dimension
        margin_y = min(margin_px, height // 2)
        margin_x = min(margin_px, width // 2)

        # Build 1D ramp profiles for each axis
        if mode == "gaussian":
            # Gaussian ramp: smooth falloff
            ramp_y = _gaussian_ramp(height, margin_y)
            ramp_x = _gaussian_ramp(width, margin_x)
        else:
            # Linear ramp
            ramp_y = _linear_ramp(height, margin_y)
            ramp_x = _linear_ramp(width, margin_x)

        # Outer product to get 2D weight
        weight = np.outer(ramp_y, ramp_x)
        return weight

    def upscale_frames(
        self,
        frames: list,
        progress_callback=None,
    ) -> list:
        """Upscale a batch of frames.

        Args:
            frames: List of numpy arrays (BGR uint8)
            progress_callback: Optional callback(current, total)

        Returns:
            List of upscaled numpy arrays
        """
        self._load_model()
        results = []
        total = len(frames)

        for i, frame in enumerate(frames):
            output, _ = self._upsampler.enhance(frame, outscale=self._outscale)
            results.append(output)
            if progress_callback:
                progress_callback(i + 1, total)

        return results

    @staticmethod
    def upscale_via_ffmpeg(
        input_path: str,
        output_path: str,
        scale: int = 2,
        preset: str = "slow",
        crf: int = 18,
    ) -> str:
        """Fallback: upscale video using ffmpeg's built-in scalers.

        Uses lanczos scaling for best quality without ML models.

        Args:
            input_path: Input video path
            output_path: Output video path
            scale: Upscale factor (2 or 4)
            preset: x264 encoding preset
            crf: Constant rate factor (lower = better quality)

        Returns:
            Output path
        """
        import subprocess

        vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg upscale failed: {result.stderr[:300]}")

        print(f"[Upscale] ffmpeg lanczos {scale}× upscale complete")
        return output_path


def _linear_ramp(length: int, margin: int) -> np.ndarray:
    """Create a 1D linear ramp: 0 at edges, 1 in center.

    Args:
        length: Total length of the ramp
        margin: Width of the ramp-up/ramp-down region

    Returns:
        1D array of shape (length,) with values in [0, 1]
    """
    ramp = np.ones(length, dtype=np.float64)
    for i in range(margin):
        val = (i + 1) / margin
        ramp[i] = val
        ramp[length - 1 - i] = val
    return ramp


def _gaussian_ramp(length: int, margin: int) -> np.ndarray:
    """Create a 1D Gaussian-like smooth ramp: near 0 at edges, 1 in center.

    Uses a cosine-based smoothstep (Hanning-like) which is computationally
    cheap and produces very smooth blending without harsh transitions.

    Args:
        length: Total length of the ramp
        margin: Width of the ramp-up/ramp-down region

    Returns:
        1D array of shape (length,) with values in [0, 1]
    """
    ramp = np.ones(length, dtype=np.float64)
    for i in range(margin):
        # Cosine smoothstep: 0.5 * (1 - cos(pi * t)) where t in [0, 1]
        t = (i + 1) / margin
        val = 0.5 * (1.0 - np.cos(np.pi * t))
        ramp[i] = val
        ramp[length - 1 - i] = val
    return ramp


def create_upscaler(
    scale: int = 2,
    model_name: str | None = None,
    device: str | None = None,
) -> PixelUpscaler | None:
    """Factory: create upscaler with graceful fallback.

    Returns PixelUpscaler if realesrgan is installed, None otherwise.
    """
    try:
        return PixelUpscaler(scale=scale, model_name=model_name, device=device)
    except ImportError:
        print("[Upscale] realesrgan not installed, upscaling will be skipped")
        return None
