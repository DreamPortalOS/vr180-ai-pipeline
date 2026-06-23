"""Tests for VR180 Prompt Builder (pipeline/prompt_builder.py).

Verifies:
- Different scene_type values return correct templates
- User original content is preserved
- Positive prompt contains key VR180 constraint keywords
- Negative prompt contains anti-nausea exclusion terms
- Edge cases (whitespace, unknown scene types, etc.)
"""

import pytest

from pipeline.prompt_builder import wrap_prompt_for_vr180

# ---------------------------------------------------------------------------
# Scene-type template tests
# ---------------------------------------------------------------------------


class TestSceneTypes:
    """Verify that each scene_type produces the correct motion template."""

    @pytest.mark.parametrize(
        "scene_type",
        ["fpv", "walkthrough", "orbit", "static"],
    )
    def test_returns_dict_with_positive_and_negative(self, scene_type: str):
        result = wrap_prompt_for_vr180("A beautiful landscape", scene_type=scene_type)
        assert isinstance(result, dict)
        assert "positive" in result
        assert "negative" in result
        assert isinstance(result["positive"], str)
        assert isinstance(result["negative"], str)

    def test_fpv_contains_fpv_keywords(self):
        result = wrap_prompt_for_vr180("Dinosaurs in a valley", scene_type="fpv")
        pos = result["positive"].lower()
        assert "fpv" in pos
        assert "forward motion" in pos or "continuous forward" in pos
        assert "8k" in pos

    def test_walkthrough_contains_walking_pace(self):
        result = wrap_prompt_for_vr180("A museum tour", scene_type="walkthrough")
        pos = result["positive"].lower()
        assert "walkthrough" in pos
        assert "walking pace" in pos

    def test_walkthrough_negative_excludes_running(self):
        result = wrap_prompt_for_vr180("A museum tour", scene_type="walkthrough")
        neg = result["negative"].lower()
        assert "running" in neg
        assert "teleporting" in neg

    def test_orbit_contains_orbital_motion(self):
        result = wrap_prompt_for_vr180("A statue", scene_type="orbit")
        pos = result["positive"].lower()
        assert "orbital" in pos
        assert "circular motion" in pos

    def test_orbit_negative_excludes_erratic(self):
        result = wrap_prompt_for_vr180("A statue", scene_type="orbit")
        neg = result["negative"].lower()
        assert "erratic orbit" in neg

    def test_static_contains_locked_camera(self):
        result = wrap_prompt_for_vr180("A still life", scene_type="static")
        pos = result["positive"].lower()
        assert "static camera" in pos or "locked" in pos
        assert "no camera movement" in pos

    def test_static_negative_excludes_movement(self):
        result = wrap_prompt_for_vr180("A still life", scene_type="static")
        neg = result["negative"].lower()
        assert "camera movement" in neg
        assert "panning" in neg


# ---------------------------------------------------------------------------
# User content preservation
# ---------------------------------------------------------------------------


class TestUserContentPreservation:
    """Verify the user's original prompt text is never modified or removed."""

    def test_user_text_is_prefix_of_positive(self):
        user_text = "A herd of brachiosaurus grazing in a lush valley"
        result = wrap_prompt_for_vr180(user_text, scene_type="fpv")
        assert result["positive"].startswith(user_text)

    def test_user_text_with_special_chars_preserved(self):
        user_text = "Camera flies over ~100° FOV, 8K HDR @ golden hour!"
        result = wrap_prompt_for_vr180(user_text, scene_type="fpv")
        assert user_text in result["positive"]

    def test_user_text_not_altered(self):
        user_text = "Dinosaurs eating leaves from tall trees"
        result = wrap_prompt_for_vr180(user_text)
        # The exact user text must appear as a contiguous substring
        assert user_text in result["positive"]

    def test_long_user_prompt_preserved(self):
        user_text = "First-person FPV drone flight gliding smoothly through a prehistoric valley " * 3
        result = wrap_prompt_for_vr180(user_text, scene_type="fpv")
        # The function strips trailing whitespace, so compare against stripped version
        assert user_text.strip() in result["positive"]


# ---------------------------------------------------------------------------
# Key constraint keywords in positive prompt
# ---------------------------------------------------------------------------


class TestPositiveConstraints:
    """Verify positive prompt contains VR180-critical keywords."""

    @pytest.fixture(params=["fpv", "walkthrough", "orbit", "static"])
    def result(self, request):
        return wrap_prompt_for_vr180("A scenic view", scene_type=request.param)

    def test_contains_8k(self, result):
        assert "8K" in result["positive"]

    def test_contains_photorealistic(self, result):
        assert "photorealistic" in result["positive"]

    def test_contains_depth_layers(self, result):
        pos = result["positive"]
        assert "depth layers" in pos or "foreground" in pos

    def test_contains_sharp_focus(self, result):
        assert "sharp focus" in result["positive"]


class TestStableHorizon:
    """Verify stable horizon constraint (excluding static scene type)."""

    @pytest.fixture(params=["fpv", "walkthrough", "orbit"])
    def result(self, request):
        return wrap_prompt_for_vr180("A scenic view", scene_type=request.param)

    def test_contains_stable_horizon(self, result):
        pos = result["positive"].lower()
        assert "horizon" in pos or "level" in pos


# ---------------------------------------------------------------------------
# Negative prompt anti-nausea exclusions
# ---------------------------------------------------------------------------


class TestNegativeExclusions:
    """Verify negative prompt excludes VR180-averse terms."""

    @pytest.fixture(params=["fpv", "walkthrough", "orbit", "static"])
    def result(self, request):
        return wrap_prompt_for_vr180("A scenic view", scene_type=request.param)

    def test_negative_excludes_camera_shake(self, result):
        assert "camera shake" in result["negative"]

    def test_negative_excludes_motion_blur(self, result):
        assert "motion blur" in result["negative"]

    def test_negative_excludes_flat_composition(self, result):
        assert "flat composition" in result["negative"]

    def test_fpv_negative_excludes_barrel_rolls(self):
        result = wrap_prompt_for_vr180("Flight", scene_type="fpv")
        assert "barrel rolls" in result["negative"]

    def test_fpv_negative_excludes_rapid_turns(self):
        result = wrap_prompt_for_vr180("Flight", scene_type="fpv")
        assert "rapid turns" in result["negative"]


# ---------------------------------------------------------------------------
# Edge cases & unknown scene types
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty prompt, unknown scene_type, whitespace."""

    def test_empty_prompt_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            wrap_prompt_for_vr180("")

    def test_whitespace_only_prompt_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            wrap_prompt_for_vr180("   ")

    def test_none_prompt_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            wrap_prompt_for_vr180(None)

    def test_unknown_scene_type_falls_back_to_fpv(self):
        result = wrap_prompt_for_vr180("A scene", scene_type="aerial")
        fpv_result = wrap_prompt_for_vr180("A scene", scene_type="fpv")
        assert result["positive"] == fpv_result["positive"]
        assert result["negative"] == fpv_result["negative"]

    def test_scene_type_case_insensitive(self):
        result_lower = wrap_prompt_for_vr180("Test", scene_type="fpv")
        result_upper = wrap_prompt_for_vr180("Test", scene_type="FPV")
        assert result_lower == result_upper

    def test_scene_type_with_whitespace(self):
        result = wrap_prompt_for_vr180("Test", scene_type="  orbit  ")
        assert "orbital" in result["positive"].lower()

    def test_default_scene_type_is_fpv(self):
        result_default = wrap_prompt_for_vr180("Test")
        result_fpv = wrap_prompt_for_vr180("Test", scene_type="fpv")
        assert result_default == result_fpv

    def test_positive_longer_than_user_prompt(self):
        user_text = "Short"
        result = wrap_prompt_for_vr180(user_text)
        assert len(result["positive"]) > len(user_text)

    def test_negative_not_empty(self):
        for scene in ("fpv", "walkthrough", "orbit", "static"):
            result = wrap_prompt_for_vr180("Test", scene_type=scene)
            assert len(result["negative"]) > 0
