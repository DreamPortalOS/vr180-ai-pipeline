"""Tests for Prompt Lab (scripts/prompt_lab.py).

Verifies:
- Manifest structure (each entry has target, scene, positive, negative, notes)
- Correct number of variants = len(targets) * len(scenes)
- All four target fields present in every entry
- Edge cases: empty prompt raises ValueError
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from scripts.prompt_lab import DEFAULT_SCENES, DEFAULT_TARGETS, generate_manifest


class TestGenerateManifest:
    """Core manifest generation tests."""

    def test_manifest_structure(self) -> None:
        """Each entry must contain all four required fields."""
        manifest = generate_manifest("A dragon flying through clouds")
        assert len(manifest) > 0
        for item in manifest:
            assert "target" in item
            assert "scene" in item
            assert "positive" in item
            assert "negative" in item
            assert "notes" in item
            assert isinstance(item["positive"], str)
            assert isinstance(item["negative"], str)
            assert isinstance(item["notes"], str)

    def test_default_variant_count(self) -> None:
        """Default targets × scenes = 3 × 3 = 9 variants."""
        manifest = generate_manifest("A test prompt")
        expected = len(DEFAULT_TARGETS) * len(DEFAULT_SCENES)
        assert len(manifest) == expected, f"Expected {expected} variants, got {len(manifest)}"

    def test_custom_targets_count(self) -> None:
        """Custom targets × default scenes = 2 × 3 = 6."""
        targets = ["vr180_flight", "fulldome_180"]
        manifest = generate_manifest(
            "A test prompt",
            targets=targets,
        )
        expected = len(targets) * len(DEFAULT_SCENES)
        assert len(manifest) == expected

    def test_custom_scenes_count(self) -> None:
        """Default targets × custom scenes = 3 × 2 = 6."""
        scenes = ["fpv", "static"]
        manifest = generate_manifest(
            "A test prompt",
            scenes=scenes,
        )
        expected = len(DEFAULT_TARGETS) * len(scenes)
        assert len(manifest) == expected

    def test_custom_both_count(self) -> None:
        """Custom targets × custom scenes = 2 × 2 = 4."""
        targets = ["vr180_flight", "vr360_dome"]
        scenes = ["orbit", "static"]
        manifest = generate_manifest(
            "A test prompt",
            targets=targets,
            scenes=scenes,
        )
        expected = len(targets) * len(scenes)
        assert len(manifest) == expected

    def test_single_variant(self) -> None:
        """Single target × single scene = 1 variant."""
        manifest = generate_manifest(
            "A test prompt",
            targets=["vr180_flight"],
            scenes=["fpv"],
        )
        assert len(manifest) == 1
        item = manifest[0]
        assert item["target"] == "vr180_flight"
        assert item["scene"] == "fpv"
        assert len(item["positive"]) > 0
        assert len(item["negative"]) > 0

    def test_user_prompt_preserved(self) -> None:
        """User's original prompt must appear verbatim in every positive."""
        user_text = "A lone astronaut floating in the void"
        manifest = generate_manifest(user_text)
        for item in manifest:
            assert user_text in item["positive"], f"Prompt not preserved in {item['target']}/{item['scene']}"

    def test_fulldome_notes_empty(self) -> None:
        """fulldome_180 entries should have empty notes (no special notes)."""
        manifest = generate_manifest(
            "Test",
            targets=["fulldome_180"],
            scenes=["fpv"],
        )
        assert manifest[0]["notes"] == ""

    def test_vr360_dome_notes_not_empty(self) -> None:
        """vr360_dome entries should have non-empty notes."""
        manifest = generate_manifest(
            "Test",
            targets=["vr360_dome"],
            scenes=["static"],
        )
        assert manifest[0]["notes"] != ""

    def test_empty_prompt_raises_value_error(self) -> None:
        """Empty or whitespace-only prompt must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            generate_manifest("")
        with pytest.raises(ValueError, match="non-empty"):
            generate_manifest("   ")

    def test_targets_are_distinct_across_variants(self) -> None:
        """Each variant should have a unique target/scene combination."""
        manifest = generate_manifest("Distinct test")
        combos = {(item["target"], item["scene"]) for item in manifest}
        assert len(combos) == len(manifest), "Duplicate target/scene combinations detected"


class TestManifestSerialization:
    """Verify the manifest can be serialized and deserialized cleanly."""

    def test_manifest_json_roundtrip(self) -> None:
        """Generate manifest, serialize to JSON, deserialize, verify structure."""
        original = generate_manifest("Roundtrip test")
        json_str = json.dumps(original, indent=2, ensure_ascii=False)
        restored = json.loads(json_str)
        assert len(restored) == len(original)
        for orig_item, rest_item in zip(original, restored, strict=False):
            assert orig_item == rest_item

    def test_manifest_serializable_to_file(self) -> None:
        """Manifest must be writable to a file and readable back."""
        original = generate_manifest("File serialization test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(original, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            restored = json.loads(path.read_text(encoding="utf-8"))
        assert len(restored) == len(original)
        assert restored == original
