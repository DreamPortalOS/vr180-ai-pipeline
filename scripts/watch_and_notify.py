#!/usr/bin/env python3
"""
Hermes Watch & Notify Agent

Automatically watches a directory for new VR180 video output files and
sends a notification via Feishu (飞书) webhook when a new file is detected.

Usage:
    # One-shot: notify about the latest VR180 file
    FEISHU_WEBHOOK_URL="https://..." python scripts/watch_and_notify.py --once

    # Continuous watch mode (polling, no watchdog dependency)
    FEISHU_WEBHOOK_URL="https://..." python scripts/watch_and_notify.py --watch-dir video/

    # After pipeline run (add to run_pipeline.py or call manually)
    python scripts/watch_and_notify.py --file video/output_vr180.mp4 --status "✅ 完成"
"""

import argparse
import logging
import os
import sys
import time

# Add project root to sys.path so `notifications` is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from notifications.feishu import send_vr180_notification  # noqa: E402

logger = logging.getLogger("hermes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_vr180(watch_dir: str, pattern: str = ".mp4") -> str | None:
    """Return the path of the most recently modified file in *watch_dir*."""
    candidates: list[str] = []
    try:
        for entry in os.scandir(watch_dir):
            if entry.is_file() and entry.name.endswith(pattern):
                candidates.append(entry.path)
    except FileNotFoundError:
        logger.error("Watch directory not found: %s", watch_dir)
        return None

    if not candidates:
        return None

    # Most recently modified first
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _notify(file_path: str, status: str = "✅ 制作完成") -> bool:
    """Notify via Feishu and log result."""
    logger.info("Sending notification for: %s", file_path)
    ok = send_vr180_notification(file_path, status=status)
    if ok:
        logger.info("Notification sent successfully.")
    else:
        logger.warning("Notification failed (check FEISHU_WEBHOOK_URL).")
    return ok


# ---------------------------------------------------------------------------
# Watch modes
# ---------------------------------------------------------------------------


def notify_once(watch_dir: str = "video/", pattern: str = ".mp4") -> None:
    """Find the latest VR180 file and send one notification."""
    latest = find_latest_vr180(watch_dir, pattern)
    if latest is None:
        logger.info("No VR180 files found in %s", watch_dir)
        sys.exit(0)
    _notify(latest)


def notify_file(file_path: str, status: str = "✅ 制作完成") -> None:
    """Send notification for a specific file."""
    if not os.path.isfile(file_path):
        logger.error("Specified file does not exist: %s", file_path)
        sys.exit(1)
    _notify(file_path, status=status)


def watch_forever(
    watch_dir: str,
    pattern: str = ".mp4",
    poll_interval: int = 30,
) -> None:
    """Poll *watch_dir* and notify on new/changed VR180 files.

    Uses simple file-mtime polling (no external file-watch library) so
    there is zero extra dependency.
    """
    logger.info(
        "Hermes Agent — watching %s (pattern=%s, poll=%ds)",
        watch_dir,
        pattern,
        poll_interval,
    )

    last_mtimes: dict[str, float] = {}
    while True:
        try:
            for entry in os.scandir(watch_dir):
                if not entry.is_file() or not entry.name.endswith(pattern):
                    continue
                mtime = entry.stat().st_mtime
                prev = last_mtimes.get(entry.path)
                if prev is None:
                    # First scan — just record, don't notify
                    last_mtimes[entry.path] = mtime
                elif mtime > prev:
                    # File was modified or re-created
                    logger.info("Detected new/changed file: %s", entry.name)
                    _notify(entry.path)
                    last_mtimes[entry.path] = mtime
        except FileNotFoundError:
            logger.warning("Watch directory %s not found; retrying...", watch_dir)
        except PermissionError as exc:
            logger.warning("Permission error scanning %s: %s", watch_dir, exc)

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hermes Agent — VR180 output file watcher + Feishu notifier",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Find latest VR180 file in watch-dir and send one notification",
    )
    mode.add_argument(
        "--file",
        type=str,
        default="",
        help="Send notification for a specific VR180 file",
    )
    mode.add_argument(
        "--watch-dir",
        type=str,
        default="",
        help="Watch a directory for new VR180 files (continuous polling)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=".mp4",
        help="File extension pattern to match (default: .mp4)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Poll interval in seconds (default: 30, used with --watch-dir)",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="✅ 制作完成",
        help="Custom status text for the Feishu card",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Dispatch
    if args.file:
        notify_file(args.file, status=args.status)
    elif args.watch_dir:
        watch_forever(args.watch_dir, pattern=args.pattern, poll_interval=args.poll_interval)
    else:
        # Default: --once with project's video/ dir
        notify_once(watch_dir=os.path.join(_PROJECT_ROOT, "video"), pattern=args.pattern)


if __name__ == "__main__":
    main()
