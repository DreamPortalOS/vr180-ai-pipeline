"""Tests for Celery tasks (mocked, no real Celery broker needed)."""

from unittest.mock import MagicMock, patch


def test_convert_task_exists():
    """Verify the convert_to_vr180 task is importable."""
    from workers.convert_tasks import convert_to_vr180

    assert convert_to_vr180 is not None


def test_convert_task_signature():
    """Verify the task name is registered correctly."""
    from workers.convert_tasks import convert_to_vr180

    assert convert_to_vr180.name == "convert.vr180"


def test_depth_only_task_exists():
    """Verify the estimate_depth_only task is importable."""
    from workers.convert_tasks import estimate_depth_only

    assert estimate_depth_only is not None
    assert estimate_depth_only.name == "convert.depth_only"


def test_upscale_task_exists():
    """Verify the upscale_video task is importable."""
    from workers.upscale_tasks import upscale_video

    assert upscale_video.name == "upscale.video"


def test_celery_app_configured():
    """Verify Celery app configuration values."""
    from workers.celery_app import app

    assert app.conf.task_serializer == "json"
    assert app.conf.result_serializer == "json"
    assert app.conf.task_track_started is True
    assert app.conf.task_acks_late is True
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.result_expires == 86400


def test_celery_app_includes():
    """Verify Celery app includes the correct task modules."""
    from workers.celery_app import app

    includes = app.conf.include or []
    assert "workers.convert_tasks" in includes
    assert "workers.upscale_tasks" in includes


def test_celery_app_accepts_json():
    """Verify Celery only accepts JSON content."""
    from workers.celery_app import app

    assert app.conf.accept_content == ["json"]


@patch("pipeline.streaming_pipeline.StreamingPipeline")
def test_convert_task_calls_pipeline(mock_pipeline_cls):
    """Test that convert task instantiates and calls StreamingPipeline."""
    mock_instance = MagicMock()
    mock_instance.process.return_value = "/tmp/output.mp4"
    mock_pipeline_cls.return_value = mock_instance

    from workers.celery_app import app
    from workers.convert_tasks import convert_to_vr180

    old_backend = app.conf.result_backend
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    app.conf.result_backend = "cache+memory://"

    try:
        result = convert_to_vr180.apply(
            kwargs={
                "input_path": "/tmp/test.mp4",
                "output_dir": "/tmp/out",
                "params": {"depth_model": "small"},
            }
        ).get()

        assert result["output_path"] == "/tmp/output.mp4"
        assert result["metadata"]["input"] == "/tmp/test.mp4"
        assert result["metadata"]["params"]["depth_model"] == "small"
        mock_instance.process.assert_called_once()
    finally:
        app.conf.task_always_eager = False
        app.conf.task_eager_propagates = False
        app.conf.result_backend = old_backend


@patch("pipeline.streaming_pipeline.StreamingPipeline")
def test_convert_task_default_params(mock_pipeline_cls):
    """Test that convert task uses default params when not provided."""
    mock_instance = MagicMock()
    mock_instance.process.return_value = "/tmp/output.mp4"
    mock_pipeline_cls.return_value = mock_instance

    from workers.celery_app import app
    from workers.convert_tasks import convert_to_vr180

    old_backend = app.conf.result_backend
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    app.conf.result_backend = "cache+memory://"

    try:
        convert_to_vr180.apply(
            kwargs={
                "input_path": "/tmp/test.mp4",
                "output_dir": "/tmp/out",
                "params": {},
            }
        ).get()

        mock_pipeline_cls.assert_called_once_with(depth_model="small")
    finally:
        app.conf.task_always_eager = False
        app.conf.task_eager_propagates = False
        app.conf.result_backend = old_backend
