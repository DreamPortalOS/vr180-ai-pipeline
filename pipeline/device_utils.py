"""
Device Detection Utilities (PRD §7.3)
Automatically detects the best available compute device:
  1. NVIDIA CUDA (with VRAM reporting)
  2. Apple Silicon MPS
  3. CPU fallback
"""

import logging

log = logging.getLogger("vr180-device")


def detect_best_device() -> str:
    """Auto-detect the best available compute device.

    Priority:
      1. CUDA (NVIDIA GPU with VRAM ≥ 4 GB)
      2. MPS  (Apple Silicon, macOS 13+)
      3. CPU  (universal fallback)

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    try:
        import torch

        # --- CUDA ---
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_mem / 1e9
            log.info(f"🟢 CUDA detected: {gpu_name} ({vram_gb:.0f} GB VRAM)")
            if vram_gb < 4.0:
                log.warning(
                    f"⚠️  Low VRAM ({vram_gb:.1f} GB). "
                    "Consider --upscale-ffmpeg or smaller tile-size."
                )
            return "cuda"

        # --- MPS ---
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import platform
            chip = platform.processor() or platform.machine()
            log.info(f"🟢 MPS detected: Apple {chip}")
            return "mps"

    except ImportError:
        log.warning("PyTorch not installed — device auto-detect limited to CPU.")

    log.info("🟡 No GPU detected — falling back to CPU (will be slow).")
    return "cpu"


def get_device_info() -> dict:
    """Return a dict describing the currently detected device.

    Keys:
        device: 'cuda', 'mps', or 'cpu'
        name:   Human-readable device name (e.g. 'Apple M4 Max', 'NVIDIA RTX 4090', 'CPU')

    Returns:
        Dict with at least 'device' and 'name' keys.
    """
    dev = detect_best_device()
    name = "CPU"

    try:
        import torch
        if dev == "cuda":
            name = torch.cuda.get_device_name(0)
        elif dev == "mps":
            import platform
            name = f"Apple {platform.processor() or platform.machine()}"
    except ImportError:
        pass

    return {"device": dev, "name": name}


def resolve_device(device: str) -> str:
    """Validate and normalise a user-specified device string.

    Args:
        device: User-provided device string (e.g. "cuda", "mps", "cpu",
                "cuda:0", or None/empty).

    Returns:
        Normalised device string.

    Raises:
        ValueError: If the requested device is not available.
    """
    if not device:
        return detect_best_device()

    device = device.strip().lower()

    # Allow "cuda:0", "cuda:1", etc.
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA requested but not available. "
                    "Install PyTorch with CUDA support or use --device mps/cpu."
                )
            # Validate device index if specified
            if ":" in device:
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    raise ValueError(
                        f"CUDA device {idx} not found. "
                        f"Available: {torch.cuda.device_count()} device(s)."
                    )
            return device
        except ImportError:
            raise ValueError("PyTorch not installed — cannot use CUDA.")

    if device == "mps":
        try:
            import torch
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise ValueError(
                    "MPS requested but not available. "
                    "Requires Apple Silicon + macOS 13+ + PyTorch 2.0+."
                )
            return device
        except ImportError:
            raise ValueError("PyTorch not installed — cannot use MPS.")

    if device == "cpu":
        return device

    raise ValueError(f"Unknown device: '{device}'. Use cuda, mps, or cpu.")