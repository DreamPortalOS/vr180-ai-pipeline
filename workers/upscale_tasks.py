"""Celery tasks for video upscaling."""

import logging

from workers.celery_app import app

log = logging.getLogger(__name__)


@app.task(bind=True, name="upscale.video", max_retries=1)
def upscale_video(self, input_path: str, output_path: str, factor: int = 2) -> dict:
    """Upscale a video file using the pipeline upscaler.

    Args:
        input_path: Path to the source video file.
        output_path: Path where the upscaled video will be written.
        factor: Upscale factor, 2 or 4.

    Returns:
        dict with output_path key.
    """
    from pipeline.upscaler import Upscaler

    self.update_state(state="STARTED", meta={"stage": "upscaling", "progress": 0})
    upscaler = Upscaler(scale_factor=factor)
    upscaler.upscale_video(
        input_path=input_path,
        output_path=output_path,
        progress_callback=lambda pct: self.update_state(state="PROGRESS", meta={"stage": "upscaling", "progress": pct}),
    )
    return {"output_path": output_path}
