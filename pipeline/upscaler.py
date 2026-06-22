"""
Stage 0 / Stage 3.5 — Pixel Upscaling
Super-resolution using Real-ESRGAN for enhanced visual quality.
Supports 2× and 4× upscaling with MPS/CUDA/CPU auto-detection.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


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

    MODELS = {
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
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        half_precision: bool = False,
    ):
        self.scale = scale
        self.model_name = model_name or f"RealESRGAN_x{scale}plus"
        self.device = self._resolve_device(device)
        self.half = half_precision and self.device == "cuda"
        self._upsampler = None

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
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
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError as e:
            raise ImportError(
                "Real-ESRGAN not installed. Run: "
                "pip install realesrgan basicsr\n"
                f"Original error: {e}"
            )

        model_info = self.MODELS.get(self.model_name)
        if not model_info:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(self.MODELS.keys())}"
            )

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

        # Real-ESRGAN expects BGR uint8
        output, _ = self._upsampler.enhance(frame, outscale=self._outscale)
        return output

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
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg upscale failed: {result.stderr[:300]}")

        print(f"[Upscale] ffmpeg lanczos {scale}× upscale complete")
        return output_path


def create_upscaler(
    scale: int = 2,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[PixelUpscaler]:
    """Factory: create upscaler with graceful fallback.

    Returns PixelUpscaler if realesrgan is installed, None otherwise.
    """
    try:
        return PixelUpscaler(scale=scale, model_name=model_name, device=device)
    except ImportError:
        print("[Upscale] realesrgan not installed, upscaling will be skipped")
        return None
