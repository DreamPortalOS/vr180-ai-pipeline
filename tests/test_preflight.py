"""Tests for the resource preflight check (issue #226).

All memory/VRAM reads are injected via monkeypatch callables so the tests
are deterministic, CPU-only, and do not depend on the host's real resources.
"""

import pytest

from pipeline.device_utils import (
    format_preflight,
    preflight_check,
)

RAM_GB = 1024**3


def test_enough_ram_and_vram_passes():
    report = preflight_check(
        min_free_ram_gb=4.0,
        min_free_vram_gb=2.0,
        _get_free_ram=lambda: 16 * RAM_GB,
        _get_free_vram=lambda: 8 * RAM_GB,
    )
    assert report.ok is True
    assert report.reasons == []
    assert report.free_ram_gb == pytest.approx(16.0, abs=1e-9)
    assert report.free_vram_gb == pytest.approx(8.0, abs=1e-9)


def test_ram_shortage_fails_with_values_and_threshold():
    report = preflight_check(
        min_free_ram_gb=8.0,
        _get_free_ram=lambda: 4 * RAM_GB,
        _get_free_vram=lambda: 8 * RAM_GB,
    )
    assert report.ok is False
    assert len(report.reasons) == 1
    reason = report.reasons[0]
    assert "4.0" in reason and "8.0" in reason
    assert "RAM" in reason


def test_vram_shortage_fails_with_values_and_threshold():
    report = preflight_check(
        min_free_ram_gb=4.0,
        min_free_vram_gb=8.0,
        _get_free_ram=lambda: 16 * RAM_GB,
        _get_free_vram=lambda: 4 * RAM_GB,
    )
    assert report.ok is False
    assert len(report.reasons) == 1
    reason = report.reasons[0]
    assert "4.0" in reason and "8.0" in reason
    assert "VRAM" in reason


def test_no_cuda_sets_vram_to_none_and_does_not_affect_ok():
    # vram callable returns None -> no CUDA; RAM is sufficient -> ok=True.
    report = preflight_check(
        min_free_ram_gb=4.0,
        min_free_vram_gb=8.0,  # would fail if checked, but CUDA is absent
        _get_free_ram=lambda: 16 * RAM_GB,
        _get_free_vram=lambda: None,
    )
    assert report.ok is True
    assert report.free_vram_gb is None
    assert report.reasons == []


def test_no_cuda_still_fails_when_ram_short():
    report = preflight_check(
        min_free_ram_gb=16.0,
        min_free_vram_gb=2.0,
        _get_free_ram=lambda: 4 * RAM_GB,
        _get_free_vram=lambda: None,
    )
    assert report.ok is False
    assert report.free_vram_gb is None
    assert len(report.reasons) == 1
    assert "VRAM" not in report.reasons[0]
    assert "RAM" in report.reasons[0]


def test_format_preflight_contains_both_values():
    report = preflight_check(
        min_free_ram_gb=4.0,
        min_free_vram_gb=2.0,
        _get_free_ram=lambda: 16 * RAM_GB,
        _get_free_vram=lambda: 8 * RAM_GB,
    )
    line = format_preflight(report)
    assert "RAM 16.0 GB" in line
    assert "VRAM 8.0 GB" in line
    assert "OK" in line


def test_format_preflight_no_cuda_shows_na():
    report = preflight_check(
        min_free_ram_gb=4.0,
        _get_free_ram=lambda: 16 * RAM_GB,
        _get_free_vram=lambda: None,
    )
    line = format_preflight(report)
    assert "RAM 16.0 GB" in line
    assert "no CUDA" in line


def test_format_preflight_failure_includes_reason():
    report = preflight_check(
        min_free_ram_gb=16.0,
        _get_free_ram=lambda: 4 * RAM_GB,
        _get_free_vram=lambda: None,
    )
    line = format_preflight(report)
    assert "FAIL" in line
    assert "reasons" in line
    assert "4.0" in line
    assert "16.0" in line
