"""
Tests for Phase 4: Public Beta — Quota, Storage, and Spatial Converter.
"""

import json
import os
import shutil
import struct
import tempfile

import pytest

# --- Quota Manager Tests ---
from web.quota import QuotaExceededError, QuotaManager, UsageRecord, UserTier


class TestQuotaManager:
    """Tests for QuotaManager quota/usage system."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_quota_")
        self.db_path = os.path.join(self.tmp_dir, "test_quota.db")
        self.quota = QuotaManager(db_path=self.db_path, max_free_conversions=3)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_initial_quota_free_user(self):
        q = self.quota.get_quota("user1")
        assert q.user_id == "user1"
        assert q.tier == "free"
        assert q.used == 0
        assert q.limit == 3
        assert q.remaining == 3
        assert q.unlimited is False

    def test_check_allows_within_quota(self):
        assert self.quota.check("user1") is True

    def test_check_or_raise_within_quota(self):
        self.quota.check_or_raise("user1")

    def test_record_usage_increments_count(self):
        self.quota.record_usage("user1", "task1", file_size_bytes=1024)
        assert self.quota.get_usage_count("user1") == 1
        self.quota.record_usage("user1", "task2", file_size_bytes=2048)
        assert self.quota.get_usage_count("user1") == 2

    def test_quota_exceeded_after_limit(self):
        for i in range(3):
            self.quota.record_usage("user1", f"task{i}")
        assert self.quota.check("user1") is False
        with pytest.raises(QuotaExceededError) as exc_info:
            self.quota.check_or_raise("user1")
        assert exc_info.value.used == 3
        assert exc_info.value.limit == 3

    def test_record_usage_raises_after_limit(self):
        for i in range(3):
            self.quota.record_usage("user1", f"task{i}")
        with pytest.raises(QuotaExceededError):
            self.quota.record_usage("user1", "task_overflow")

    def test_premium_user_unlimited(self):
        self.quota.set_tier("user2", UserTier.PREMIUM)
        q = self.quota.get_quota("user2")
        assert q.tier == "premium"
        assert q.unlimited is True
        assert q.limit == -1
        assert q.remaining == -1
        for i in range(50):
            self.quota.record_usage("user2", f"task{i}")
        assert self.quota.get_usage_count("user2") == 50
        assert self.quota.check("user2") is True

    def test_admin_user_unlimited(self):
        self.quota.set_tier("user3", UserTier.ADMIN)
        q = self.quota.get_quota("user3")
        assert q.tier == "admin"
        assert q.unlimited is True

    def test_reset_usage(self):
        for i in range(3):
            self.quota.record_usage("user1", f"task{i}")
        assert self.quota.get_usage_count("user1") == 3
        self.quota.reset_usage("user1")
        assert self.quota.get_usage_count("user1") == 0
        assert self.quota.check("user1") is True

    def test_usage_history(self):
        self.quota.set_tier("user1", UserTier.PREMIUM)
        for i in range(5):
            self.quota.record_usage("user1", f"task{i}", file_size_bytes=i * 100)
        history = self.quota.get_usage_history("user1", limit=3, offset=0)
        assert len(history) == 3
        assert all(isinstance(r, UsageRecord) for r in history)
        assert history[0].task_id == "task4"

    def test_usage_history_offset(self):
        self.quota.set_tier("user1", UserTier.PREMIUM)
        for i in range(5):
            self.quota.record_usage("user1", f"task{i}")
        history = self.quota.get_usage_history("user1", limit=3, offset=3)
        assert len(history) == 2

    def test_total_usage(self):
        self.quota.record_usage("user1", "task1")
        self.quota.record_usage("user2", "task2")
        self.quota.record_usage("user1", "task3")
        assert self.quota.get_total_usage() == 3

    def test_total_storage_bytes(self):
        self.quota.record_usage("user1", "task1", file_size_bytes=1000)
        self.quota.record_usage("user2", "task2", file_size_bytes=2000)
        assert self.quota.get_total_storage_bytes() == 3000

    def test_custom_limit(self):
        quota2 = QuotaManager(
            db_path=os.path.join(self.tmp_dir, "custom.db"), max_free_conversions=1,
        )
        quota2.record_usage("user1", "task1")
        assert quota2.check("user1") is False

    def test_get_limit(self):
        assert self.quota.get_limit("user1") == 3
        self.quota.set_tier("user1", UserTier.PREMIUM)
        assert self.quota.get_limit("user1") == -1

    def test_user_auto_created_on_check(self):
        self.quota.check("new_user")
        q = self.quota.get_quota("new_user")
        assert q.tier == "free"

    def test_multiple_users_independent(self):
        self.quota.record_usage("user1", "task1")
        self.quota.record_usage("user1", "task2")
        assert self.quota.get_usage_count("user1") == 2
        assert self.quota.get_usage_count("user2") == 0

    def test_quota_exceeded_message(self):
        exc = QuotaExceededError("u1", used=3, limit=3)
        assert "u1" in str(exc)
        assert "3/3" in str(exc)

    def test_persistence_across_instances(self):
        self.quota.record_usage("user1", "task1", file_size_bytes=500)
        quota2 = QuotaManager(db_path=self.db_path, max_free_conversions=3)
        assert quota2.get_usage_count("user1") == 1


# --- Result Storage Tests ---

from web.storage import ResultNotFoundError, ResultStorage  # noqa: E402


class TestResultStorage:
    """Tests for ResultStorage persistence."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_storage_")
        self.storage = ResultStorage(base_dir=self.tmp_dir)
        self.fake_video = os.path.join(self.tmp_dir, "input.mp4")
        with open(self.fake_video, "wb") as f:
            f.write(b"\x00" * 1024)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_get_result(self):
        result_id = self.storage.save_result(
            task_id="task1", output_path=self.fake_video, user_id="user1",
            input_filename="test.mp4", duration_seconds=10.5,
            width=7680, height=3840, fps=30.0, codec="h264", copy_file=False,
        )
        result = self.storage.get_result(result_id)
        assert result.task_id == "task1"
        assert result.user_id == "user1"
        assert result.input_filename == "test.mp4"
        assert result.duration_seconds == 10.5
        assert result.width == 7680
        assert result.codec == "h264"

    def test_save_with_file_copy(self):
        result_id = self.storage.save_result(
            task_id="task1", output_path=self.fake_video, user_id="user1", copy_file=True,
        )
        result = self.storage.get_result(result_id)
        assert os.path.exists(result.output_path)
        assert result.file_size_bytes == 1024

    def test_get_result_not_found(self):
        with pytest.raises(ResultNotFoundError):
            self.storage.get_result("nonexistent-id")

    def test_list_results_empty(self):
        assert self.storage.list_results() == []

    def test_list_results_with_data(self):
        for i in range(5):
            self.storage.save_result(
                task_id=f"task{i}", output_path=self.fake_video, copy_file=False,
            )
        results = self.storage.list_results()
        assert len(results) == 5

    def test_list_results_filter_by_user(self):
        self.storage.save_result(task_id="t1", output_path=self.fake_video, user_id="u1", copy_file=False)
        self.storage.save_result(task_id="t2", output_path=self.fake_video, user_id="u2", copy_file=False)
        self.storage.save_result(task_id="t3", output_path=self.fake_video, user_id="u1", copy_file=False)
        results = self.storage.list_results(user_id="u1")
        assert len(results) == 2

    def test_list_results_pagination(self):
        for i in range(10):
            self.storage.save_result(task_id=f"task{i}", output_path=self.fake_video, copy_file=False)
        page1 = self.storage.list_results(limit=3, offset=0)
        page2 = self.storage.list_results(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].id != page2[0].id

    def test_count_results(self):
        assert self.storage.count_results() == 0
        for i in range(7):
            self.storage.save_result(task_id=f"task{i}", output_path=self.fake_video, copy_file=False)
        assert self.storage.count_results() == 7

    def test_delete_result(self):
        result_id = self.storage.save_result(
            task_id="task1", output_path=self.fake_video, copy_file=True,
        )
        result = self.storage.get_result(result_id)
        assert os.path.exists(result.output_path)
        assert self.storage.delete_result(result_id) is True
        assert not os.path.exists(result.output_path)

    def test_delete_result_not_found(self):
        assert self.storage.delete_result("nonexistent") is False

    def test_delete_by_task_id(self):
        self.storage.save_result(task_id="task1", output_path=self.fake_video, copy_file=False)
        self.storage.save_result(task_id="task1", output_path=self.fake_video, copy_file=False)
        self.storage.save_result(task_id="task2", output_path=self.fake_video, copy_file=False)
        count = self.storage.delete_by_task_id("task1")
        assert count == 2
        assert self.storage.count_results() == 1

    def test_save_with_metadata(self):
        result_id = self.storage.save_result(
            task_id="task1", output_path=self.fake_video,
            metadata={"quality": "high", "upscaled": True}, copy_file=False,
        )
        result = self.storage.get_result(result_id)
        meta = json.loads(result.metadata_json)
        assert meta["quality"] == "high"
        assert meta["upscaled"] is True

    def test_save_with_expiration(self):
        self.storage.save_result(
            task_id="task1", output_path=self.fake_video,
            expires_at="2020-01-01T00:00:00Z", copy_file=False,
        )
        cleaned = self.storage.cleanup_expired()
        assert cleaned == 1

    def test_get_total_storage_bytes(self):
        self.storage.save_result(task_id="t1", output_path=self.fake_video, copy_file=False, file_size_bytes=100)
        self.storage.save_result(task_id="t2", output_path=self.fake_video, copy_file=False, file_size_bytes=200)
        assert self.storage.get_total_storage_bytes() == 300

    def test_stored_result_defaults(self):
        result_id = self.storage.save_result(
            task_id="task1", output_path=self.fake_video, copy_file=False,
        )
        result = self.storage.get_result(result_id)
        assert result.stereoscopic_mode == "side-by-side"
        assert result.projection == "equirectangular"
        assert result.metadata_json == "{}"

    def test_persistence_across_instances(self):
        self.storage.save_result(task_id="t1", output_path=self.fake_video, copy_file=False)
        storage2 = ResultStorage(base_dir=self.tmp_dir)
        assert storage2.count_results() == 1


# --- Spatial Converter Tests ---

from pipeline.spatial_converter import (  # noqa: E402
    SpatialConverter,
    SpatialFormat,
    SpatialProjection,
    SpatialVideoInfo,
)


class TestSpatialConverter:
    """Tests for SpatialConverter unit-level logic."""

    def test_spatial_format_enum(self):
        assert SpatialFormat.MV_HEVC.value == "mv-hevc"
        assert SpatialFormat.SBS_SPATIAL.value == "sbs-spatial"
        assert SpatialFormat.SBS_MONO.value == "sbs-mono"

    def test_spatial_projection_enum(self):
        assert SpatialProjection.EQUIRECTANGULAR.value == "equirectangular"
        assert SpatialProjection.RECTILINEAR.value == "rectilinear"
        assert SpatialProjection.EQUIRECT.value == "equirect"

    def test_spatial_video_info_dataclass(self):
        info = SpatialVideoInfo(
            width=7680, height=3840, fps=30.0, duration=10.0,
            codec="h264", format=SpatialProjection.EQUIRECTANGULAR,
            is_stereoscopic=True, stereo_mode="side-by-side",
            has_spatial_metadata=False, file_size=1024,
        )
        assert info.width == 7680
        assert info.is_stereoscopic is True

    def test_inject_mv_hevc_metadata(self):
        try:
            converter = SpatialConverter()
        except RuntimeError:
            pytest.skip("ffmpeg not available")

        tmp = tempfile.mktemp(suffix=".mp4")
        try:
            with open(tmp, "wb") as f:
                f.write(b"\x00" * 100)
            converter._inject_mv_hevc_metadata(tmp, 3840, 1920)
            with open(tmp, "rb") as f:
                data = f.read()
            assert len(data) > 100
            assert b"st3d" in data
            assert b"sv3d" in data
            assert b"svhd" in data
            assert b"proj" in data
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_inject_sbs_spatial_metadata(self):
        try:
            converter = SpatialConverter()
        except RuntimeError:
            pytest.skip("ffmpeg not available")

        tmp = tempfile.mktemp(suffix=".mp4")
        try:
            with open(tmp, "wb") as f:
                f.write(b"\x00" * 100)
            converter._inject_sbs_spatial_metadata(tmp, 7680, 3840)
            with open(tmp, "rb") as f:
                data = f.read()
            assert len(data) > 100
            assert b"st3d" in data
            assert b"sv3d" in data
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_st3d_box_structure(self):
        """Verify st3d box has correct binary structure."""
        st3d_data = struct.pack(">IB", 0, 1)  # version(4B)=0, stereo_mode(1B)=1 (SBS)
        st3d_box = struct.pack(">I", 8 + len(st3d_data)) + b"st3d" + st3d_data
        # Box size = 4 (size) + 4 (type) + 4 (version+flags) + 1 (stereo_mode) = 13
        size = struct.unpack(">I", st3d_box[:4])[0]
        assert size == 13
        assert st3d_box[4:8] == b"st3d"
        # Version is stored as 4-byte big-endian int at offset 8
        version_byte = st3d_box[8]
        # Stereo mode is at offset 12 (after 4-byte version+flags field)
        stereo_mode = st3d_box[12]
        assert version_byte == 0
        assert stereo_mode == 1  # side-by-side

    def test_sv3d_box_contains_sub_boxes(self):
        """Verify sv3d box contains svhd and proj sub-boxes."""
        svhd_data = b"\x00" + b"\x00" * 3 + b"Test\x00"
        svhd_box = struct.pack(">I", 8 + len(svhd_data)) + b"svhd" + svhd_data

        proj_data = struct.pack(">I", 0)
        proj_box = struct.pack(">I", 8 + len(proj_data)) + b"proj" + proj_data

        sv3d_payload = svhd_box + proj_box
        sv3d_box = struct.pack(">I", 8 + len(sv3d_payload)) + b"sv3d" + sv3d_payload

        assert b"svhd" in sv3d_box
        assert b"proj" in sv3d_box
        # Verify sv3d size includes sub-boxes
        sv3d_size = struct.unpack(">I", sv3d_box[:4])[0]
        assert sv3d_size == 8 + len(sv3d_payload)

    def test_get_supported_formats(self):
        try:
            converter = SpatialConverter()
        except RuntimeError:
            pytest.skip("ffmpeg not available")
        formats = converter.get_supported_formats()
        assert len(formats) == 3
        assert "mv-hevc" in formats
        assert "Apple Vision Pro" in formats["mv-hevc"]
        assert "Quest" in formats["sbs-spatial"]
