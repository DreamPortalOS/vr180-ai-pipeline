"""
Feishu (飞书) notification module — Hermes Agent.

Sends rich-text card messages via Feishu bot webhook to notify users
when a new VR180 video has been produced by the pipeline.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "")
DEFAULT_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class VideoInfo:
    """Extracted metadata about a produced VR180 video file."""

    path: str
    filename: str = ""
    size_mb: float = 0.0
    duration_sec: float = 0.0
    resolution: str = ""
    codec: str = ""
    fps: float = 0.0

    def __post_init__(self) -> None:
        self.filename = os.path.basename(self.path)
        self.size_mb = self._get_file_size()
        self._probe_ffprobe()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _get_file_size(self) -> float:
        try:
            return os.path.getsize(self.path) / (1024 * 1024)
        except OSError:
            return 0.0

    def _probe_ffprobe(self) -> None:
        """Fill video metadata via ffprobe (silent on failure)."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            self.path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                return
            data = json.loads(result.stdout)

            # Duration from format
            fmt = data.get("format", {})
            dur_str = fmt.get("duration", "0")
            self.duration_sec = float(dur_str)

            # Video stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    w = stream.get("width", 0)
                    h = stream.get("height", 0)
                    if w and h:
                        self.resolution = f"{w}x{h}"
                    self.codec = stream.get("codec_name", "")
                    fps_str = stream.get("r_frame_rate", "0/1")
                    try:
                        num, den = fps_str.split("/")
                        self.fps = float(num) / float(den)
                    except (ValueError, ZeroDivisionError):
                        self.fps = 0.0
                    break
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            logger.debug("ffprobe failed for %s: %s", self.path, exc)


# ---------------------------------------------------------------------------
# Feishu card builder
# ---------------------------------------------------------------------------


def _build_vr180_card(video: VideoInfo, status: str = "✅ 制作完成") -> dict:
    """Build a Feishu interactive card JSON payload."""
    duration_str = f"{video.duration_sec:.1f}s" if video.duration_sec > 0 else "未知"
    size_str = f"{video.size_mb:.1f} MB" if video.size_mb > 0 else "未知"

    elements = [
        {
            "tag": "markdown",
            "content": (
                f"**文件：** {video.filename}\n"
                f"**状态：** {status}\n"
                f"**大小：** {size_str}\n"
                f"**时长：** {duration_str}\n"
                f"**分辨率：** {video.resolution}\n"
                f"**编码：** {video.codec}\n"
                f"**FPS：** {video.fps:.2f}"
                if video.fps > 0
                else "**FPS：** 未知"
            ),
        },
        {
            "tag": "hr",
        },
        {
            "tag": "note",
            "elements": [
                {
                    "tag": "plain_text",
                    "content": f"🕐 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · VR180 AI Pipeline",
                }
            ],
        },
    ]

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "🎬 VR180 制作通知"},
            "template": "blue",
        },
        "elements": elements,
    }

    return {"msg_type": "interactive", "card": card}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_vr180_notification(
    video_path: str,
    webhook_url: str | None = None,
    status: str = "✅ 制作完成",
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """Send a Feishu card notification about a completed VR180 video.

    Args:
        video_path: Absolute or relative path to the VR180 output file.
        webhook_url: Feishu bot webhook URL. Falls back to
            ``FEISHU_WEBHOOK_URL`` env var.
        status: Custom status message shown in the card.
        timeout: HTTP request timeout in seconds.

    Returns:
        ``True`` if the message was sent successfully, ``False`` otherwise.
    """
    url = webhook_url or DEFAULT_WEBHOOK_URL
    if not url:
        logger.error("No Feishu webhook URL configured. Set FEISHU_WEBHOOK_URL env var.")
        return False

    if not os.path.isfile(video_path):
        logger.error("Video file not found: %s", video_path)
        return False

    video = VideoInfo(video_path)
    payload = _build_vr180_card(video, status=status)

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") == 0:
                logger.info("Feishu notification sent for %s", video.filename)
                return True
            else:
                logger.error(
                    "Feishu API error: %s — %s",
                    result.get("code"),
                    result.get("msg", ""),
                )
                return False
    except httpx.HTTPError as exc:
        logger.error("Feishu webhook request failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print(f"Usage: {__file__} <video_path> [status_text]")
        sys.exit(1)

    video_path = sys.argv[1]
    status = sys.argv[2] if len(sys.argv) > 2 else "✅ 制作完成"

    success = send_vr180_notification(video_path, status=status)
    sys.exit(0 if success else 1)
