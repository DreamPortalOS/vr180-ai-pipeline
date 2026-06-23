"""Celery tasks for VR180 conversion pipeline."""
import logging
from pathlib import Path

from workers.celery_app import app

log = logging.getLogger(__name__)


@app.task(bind=True, name="convert.vr180", max_retries=2)
def convert_to_vr180(self, input_path: str, output_dir: str, params: dict) -> dict:
    """
    Full VR180 conversion pipeline task.

    params keys:
      - depth_model: "small" | "base" | "large"  (default: "small")
      - upscale_factor: 2 | 4  (default: 2)
      - outpainting: bool  (default: False)

    Returns: {"output_path": str, "metadata": dict}
    """
    from pipeline.streaming_pipeline import StreamingPipeline

    self.update_state(state="STARTED", meta={"stage": "initializing", "progress": 0})

    try:
        pipeline = StreamingPipeline(
            depth_model=params.get("depth_model", "small"),
        )

        self.update_state(
            state="PROGRESS", meta={"stage": "depth_estimation", "progress": 10}
        )

        output_path = pipeline.process(
            input_path=input_path,
            output_dir=output_dir,
            progress_callback=lambda pct, stage: self.update_state(
                state="PROGRESS", meta={"stage": stage, "progress": pct}
            ),
        )

        return {
            "output_path": str(output_path),
            "metadata": {"input": input_path, "params": params},
        }
    except Exception as exc:
        log.exception("convert_to_vr180 failed: %s", exc)
        raise self.retry(exc=exc, countdown=30)


@app.task(bind=True, name="convert.depth_only", max_retries=1)
def estimate_depth_only(
    self, input_path: str, output_dir: str, model_size: str = "small"
) -> dict:
    """Standalone depth estimation task (for preview)."""
    from pipeline.depth_estimator import DepthEstimator
    import cv2
    import numpy as np

    self.update_state(state="STARTED", meta={"stage": "loading_model", "progress": 0})

    estimator = DepthEstimator(model_size=model_size)
    cap = cv2.VideoCapture(input_path)

    depth_frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(min(total, 30)):  # preview: max 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        depth = estimator.estimate(frame)
        depth_frames.append(depth)
        self.update_state(
            state="PROGRESS",
            meta={"stage": "depth", "progress": int(i / min(total, 30) * 100)},
        )

    cap.release()

    out_path = Path(output_dir) / "depth_preview.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), np.array(depth_frames))

    return {"depth_preview_path": str(out_path), "frames_processed": len(depth_frames)}