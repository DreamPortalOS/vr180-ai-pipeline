"""
Device Detection Utilities (PRD §7.3)
Automatically detects the best available compute device:
  1. NVIDIA CUDA (with VRAM reporting)
  2. Apple Silicon MPS
  3. CPU fallback
"""

import ctypes
import logging
import platform
from dataclasses import dataclass, field

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
            vram_gb = props.total_memory / 1e9
            log.info(f"🟢 CUDA detected: {gpu_name} ({vram_gb:.0f} GB VRAM)")
            if vram_gb < 4.0:
                log.warning(f"⚠️  Low VRAM ({vram_gb:.1f} GB). Consider --upscale-ffmpeg or smaller tile-size.")
            return "cuda"

        # --- MPS ---
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
                    "CUDA requested but not available. Install PyTorch with CUDA support or use --device mps/cpu."
                )
            # Validate device index if specified
            if ":" in device:
                idx = int(device.split(":")[1])
                if idx >= torch.cuda.device_count():
                    raise ValueError(f"CUDA device {idx} not found. Available: {torch.cuda.device_count()} device(s).")
            return device
        except ImportError:
            raise ValueError("PyTorch not installed — cannot use CUDA.") from None

    if device == "mps":
        try:
            import torch

            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise ValueError("MPS requested but not available. Requires Apple Silicon + macOS 13+ + PyTorch 2.0+.")
            return device
        except ImportError:
            raise ValueError("PyTorch not installed — cannot use MPS.") from None

    if device == "cpu":
        return device

    raise ValueError(f"Unknown device: '{device}'. Use cuda, mps, or cpu.")


# --------------------------------------------------------------------------- #
# Preflight checks (issue #226) — added, no call sites wired yet.            #
# --------------------------------------------------------------------------- #


class _MEMORYSTATUSEX(ctypes.Structure):
    """Windows MEMORYSTATUSEX layout used by GlobalMemoryStatusEx."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _free_ram_bytes() -> int:
    """Read free host RAM in bytes via platform-native calls (no extra deps).

    Windows: GlobalMemoryStatusEx via ctypes Structure.
    POSIX: /proc/meminfo (MemAvailable, falling back to MemFree).
    """
    if platform.system() == "Windows":
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0
        return status.ullAvailPhys

    try:
        with open("/proc/meminfo", "rb") as f:
            for line in f:
                if line.startswith(b"MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB -> bytes
                if line.startswith(b"MemFree:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        pass
    return 0


def _free_vram_bytes() -> int | None:
    """Read free VRAM in bytes for the current CUDA device.

    Returns None when CUDA is unavailable (this is NOT a failure).
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free


@dataclass
class PreflightReport:
    """Result of a resource preflight check before launching a heavy model."""

    free_ram_gb: float
    free_vram_gb: float | None
    ok: bool
    reasons: list[str] = field(default_factory=list)


def preflight_check(
    *,
    min_free_ram_gb: float,
    min_free_vram_gb: float | None = None,
    _get_free_ram: callable = _free_ram_bytes,
    _get_free_vram: callable = _free_vram_bytes,
) -> PreflightReport:
    """Check whether host memory (and VRAM, if CUDA is available) is sufficient.

    This is a pre-launch guard: calling a heavy model when the host is memory-
    starved leads to a `MemoryError` that reads like a corrupt weights /
    diffusers bug, but whose true cause is insufficient RAM. Raising a clear
    reason up front lets the caller retry or back off instead of surfacing a
    misleading traceback.

    Args:
        min_free_ram_gb: Minimum free host RAM required (GB).
        min_free_vram_gb: Minimum free VRAM required (GB). Pass None to skip
            the VRAM check entirely (no CUDA means the check is also skipped,
            without affecting `ok`).

    Returns:
        A :class:`PreflightReport` describing current headroom and whether
        the check passed.
    """
    reasons: list[str] = []
    free_ram_bytes = _get_free_ram()
    free_ram_gb = free_ram_bytes / (1024**3)

    min_ram_bytes = int(min_free_ram_gb * (1024**3))
    if free_ram_bytes < min_ram_bytes:
        reasons.append(f"RAM 不足: 空闲 {free_ram_gb:.1f} GB < 阈值 {min_free_ram_gb:.1f} GB")

    free_vram_bytes = _get_free_vram()
    free_vram_gb = None if free_vram_bytes is None else free_vram_bytes / (1024**3)

    if free_vram_bytes is not None and min_free_vram_gb is not None:
        min_vram_bytes = int(min_free_vram_gb * (1024**3))
        if free_vram_bytes < min_vram_bytes:
            reasons.append(f"VRAM 不足: 空闲 {free_vram_gb:.1f} GB < 阈值 {min_free_vram_gb:.1f} GB")

    return PreflightReport(
        free_ram_gb=free_ram_gb,
        free_vram_gb=free_vram_gb,
        ok=len(reasons) == 0,
        reasons=reasons,
    )


def format_preflight(report: PreflightReport) -> str:
    """Format a :class:`PreflightReport` into a single log line."""
    status = "OK" if report.ok else "FAIL"
    ram = f"{report.free_ram_gb:.1f} GB"
    vram = "n/a (no CUDA)" if report.free_vram_gb is None else f"{report.free_vram_gb:.1f} GB"
    parts = [f"preflight [{status}] RAM {ram} VRAM {vram}"]
    if report.reasons:
        parts.append("reasons: " + "; ".join(report.reasons))
    return " | ".join(parts)
