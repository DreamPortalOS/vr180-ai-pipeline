"""Tests for pipeline/naming.py — scene-oriented naming + segment ids (#81)."""

from __future__ import annotations

import pytest

from pipeline.naming import (
    PRESETS,
    ROUTES,
    NamingError,
    SceneAssetSpec,
    compose_scene_name,
    parse_scene_name,
    sidecar_scene_fields,
    validate_spec,
)

# ---------------------------------------------------------------------------
# compose_scene_name
# ---------------------------------------------------------------------------


class TestCompose:
    def test_canonical_example(self):
        """The exact example from issue #81 round-trips."""
        spec = SceneAssetSpec(
            scene_id="s03",
            scene_name="santorini",
            segment_index=1,
            route="vr180",
            preset="standalone",
        )
        assert compose_scene_name(spec) == "s03_santorini_seg01_vr180_standalone.mp4"

    def test_fulldome_route(self):
        spec = SceneAssetSpec(scene_id="s05", scene_name="aurora", segment_index=2, route="fulldome", preset="pcvr")
        assert compose_scene_name(spec) == "s05_aurora_seg02_fulldome_pcvr.mp4"

    def test_custom_extension(self):
        spec = SceneAssetSpec(scene_id="s01", scene_name="cliff", segment_index=1, route="vr180", preset="source")
        assert compose_scene_name(spec, extension="mov") == "s01_cliff_seg01_vr180_source.mov"

    def test_extension_leading_dot_stripped(self):
        spec = SceneAssetSpec(scene_id="s01", scene_name="cliff", segment_index=1, route="vr180", preset="source")
        assert compose_scene_name(spec, extension=".MP4") == "s01_cliff_seg01_vr180_source.mp4"

    def test_segment_zero_pads_to_two_digits(self):
        spec = SceneAssetSpec(scene_id="s1", scene_name="x", segment_index=1, route="vr180", preset="standalone")
        assert "seg01" in compose_scene_name(spec)

    def test_segment_above_99_uses_full_width(self):
        spec = SceneAssetSpec(scene_id="s1", scene_name="x", segment_index=123, route="vr180", preset="standalone")
        assert "seg123" in compose_scene_name(spec)

    def test_none_fields_collapse_to_defaults(self):
        """None scene_id/scene_name/segment/preset get sensible defaults."""
        spec = SceneAssetSpec(route="vr180")
        name = compose_scene_name(spec)
        # Defaults: scene_id=scene, scene_name=unnamed, seg=1, preset=standalone
        assert name == "scene_unnamed_seg01_vr180_standalone.mp4"

    def test_scene_name_slugified(self):
        """Uppercase / spaces / punctuation collapse to a lowercase slug."""
        spec = SceneAssetSpec(
            scene_id="s03",
            scene_name="Santorini Coast!",
            segment_index=1,
            route="vr180",
            preset="standalone",
        )
        assert compose_scene_name(spec) == "s03_santorini_coast_seg01_vr180_standalone.mp4"

    def test_scene_name_with_underscore_preserved(self):
        spec = SceneAssetSpec(
            scene_id="s03",
            scene_name="santorini_sunset",
            segment_index=1,
            route="vr180",
            preset="standalone",
        )
        assert compose_scene_name(spec) == "s03_santorini_sunset_seg01_vr180_standalone.mp4"

    def test_scene_name_empty_after_slug_collapses_to_default(self):
        spec = SceneAssetSpec(scene_id="s03", scene_name="!!!", segment_index=1, route="vr180", preset="standalone")
        assert compose_scene_name(spec) == "s03_unnamed_seg01_vr180_standalone.mp4"

    def test_scene_id_with_dashes_and_digits(self):
        spec = SceneAssetSpec(
            scene_id="scene-1a",
            scene_name="x",
            segment_index=1,
            route="vr180",
            preset="standalone",
        )
        assert compose_scene_name(spec).startswith("scene-1a_x_seg01_")

    @pytest.mark.parametrize("bad_segment", [0, -1, -5])
    def test_non_positive_segment_raises(self, bad_segment):
        spec = SceneAssetSpec(
            scene_id="s03", scene_name="x", segment_index=bad_segment, route="vr180", preset="standalone"
        )
        with pytest.raises(NamingError, match="segment_index"):
            compose_scene_name(spec)

    @pytest.mark.parametrize("bad_id", ["S03", "S 03", " s03", "s03 ", "s.03"])
    def test_invalid_scene_id_raises(self, bad_id):
        spec = SceneAssetSpec(scene_id=bad_id, scene_name="x", segment_index=1, route="vr180", preset="standalone")
        with pytest.raises(NamingError, match="scene_id"):
            compose_scene_name(spec)

    def test_empty_scene_id_collapses_to_default(self):
        """An empty-string scene_id is treated like None → default, not an error."""
        spec = SceneAssetSpec(scene_id="", scene_name="x", segment_index=1, route="vr180", preset="standalone")
        assert compose_scene_name(spec).startswith("scene_x_seg01_")

    def test_empty_extension_raises(self):
        spec = SceneAssetSpec(scene_id="s03", scene_name="x", segment_index=1, route="vr180", preset="standalone")
        with pytest.raises(NamingError, match="extension"):
            compose_scene_name(spec, extension="")


# ---------------------------------------------------------------------------
# parse_scene_name
# ---------------------------------------------------------------------------


class TestParse:
    def test_canonical_example(self):
        spec = parse_scene_name("s03_santorini_seg01_vr180_standalone.mp4")
        assert spec == SceneAssetSpec(
            scene_id="s03",
            scene_name="santorini",
            segment_index=1,
            route="vr180",
            preset="standalone",
        )

    def test_accepts_path_only_basename(self):
        """A full path is tolerated — only the final component is parsed."""
        spec = parse_scene_name("out/scenes/s03_santorini_seg01_vr180_standalone.mp4")
        assert spec.scene_id == "s03"
        assert spec.segment_index == 1

    def test_accepts_windows_path(self):
        spec = parse_scene_name("C:\\out\\s03_santorini_seg01_vr180_standalone.mp4")
        assert spec.scene_id == "s03"

    def test_accepts_stem_without_extension(self):
        spec = parse_scene_name("s03_santorini_seg01_vr180_standalone")
        assert spec.scene_id == "s03"
        assert spec.preset == "standalone"

    def test_segment_not_zero_padded(self):
        """seg1 (no zero-pad) is accepted on parse — lenient input."""
        spec = parse_scene_name("s03_santorini_seg1_vr180_standalone.mp4")
        assert spec.segment_index == 1

    def test_segment_three_digits(self):
        spec = parse_scene_name("s03_santorini_seg123_vr180_standalone.mp4")
        assert spec.segment_index == 123

    def test_scene_name_with_underscore_rejoined(self):
        """A multi-word scene_name (with underscores) round-trips through parse."""
        spec = parse_scene_name("s03_santorini_sunset_seg01_vr180_standalone.mp4")
        assert spec.scene_name == "santorini_sunset"

    @pytest.mark.parametrize(
        "bad",
        [
            "s03_santorini_seg01_vr180.mp4",  # missing preset (4 fields)
            "s03_santorini_vr180_standalone.mp4",  # missing seg token
            "s03_seg01_vr180_standalone.mp4",  # missing scene_name (only 4 fields)
            "santorini_seg01_vr180_standalone.mp4",  # too few fields
            "s03_santorini_segXX_vr180_standalone.mp4",  # bad segment token
        ],
    )
    def test_invalid_names_raise(self, bad):
        with pytest.raises(NamingError):
            parse_scene_name(bad)

    def test_five_fields_no_extension_parses(self):
        """A stem with exactly 5 fields and no extension is valid."""
        spec = parse_scene_name("s03_santorini_seg01_vr180_standalone")
        assert spec.preset == "standalone"

    def test_bad_segment_token_raises(self):
        with pytest.raises(NamingError, match="segment token"):
            parse_scene_name("s03_santorini_seg0a_vr180_standalone.mp4")

    def test_empty_string_raises(self):
        with pytest.raises(NamingError):
            parse_scene_name("")


# ---------------------------------------------------------------------------
# Round-trip invariant
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @pytest.mark.parametrize(
        "spec",
        [
            SceneAssetSpec("s03", "santorini", 1, "vr180", "standalone"),
            SceneAssetSpec("s05", "aurora", 2, "fulldome", "pcvr"),
            SceneAssetSpec("s01", "santorini_sunset", 10, "vr180", "source"),
            SceneAssetSpec("scene-1a", "cliff_edge", 99, "fulldome", "standalone"),
            SceneAssetSpec("s9", "x", 1, "vr180", "standalone"),
        ],
    )
    def test_compose_then_parse_round_trips(self, spec):
        name = compose_scene_name(spec)
        reparsed = parse_scene_name(name)
        assert reparsed == spec

    def test_lexical_sort_matches_assembly_order(self):
        """Zero-padded segments sort lexically in assembly order — the
        property the downstream cross-fade assembler relies on."""
        specs = [SceneAssetSpec("s03", "x", i, "vr180", "standalone") for i in (1, 2, 10, 11, 20)]
        names = [compose_scene_name(s) for s in specs]
        sorted_names = sorted(names)
        # Sorted order should be ascending by segment_index.
        order = [parse_scene_name(n).segment_index for n in sorted_names]
        assert order == [1, 2, 10, 11, 20]


# ---------------------------------------------------------------------------
# validate_spec
# ---------------------------------------------------------------------------


class TestValidateSpec:
    def test_valid_spec_passes(self):
        validate_spec(SceneAssetSpec("s03", "santorini", 1, "vr180", "standalone"))  # no exception

    @pytest.mark.parametrize("route", list(ROUTES))
    def test_all_routes_valid(self, route):
        validate_spec(SceneAssetSpec("s01", "x", 1, route, "standalone"))

    @pytest.mark.parametrize("preset", list(PRESETS))
    def test_all_presets_valid(self, preset):
        validate_spec(SceneAssetSpec("s01", "x", 1, "vr180", preset))

    def test_unknown_route_raises(self):
        with pytest.raises(NamingError, match="route"):
            validate_spec(SceneAssetSpec("s01", "x", 1, "360", "standalone"))

    def test_unknown_preset_raises(self):
        with pytest.raises(NamingError, match="preset"):
            validate_spec(SceneAssetSpec("s01", "x", 1, "vr180", "ultra"))

    def test_non_positive_segment_raises(self):
        with pytest.raises(NamingError, match="segment_index"):
            validate_spec(SceneAssetSpec("s01", "x", 0, "vr180", "standalone"))

    def test_invalid_scene_id_raises(self):
        with pytest.raises(NamingError, match="scene_id"):
            validate_spec(SceneAssetSpec("S03", "x", 1, "vr180", "standalone"))

    def test_compose_does_not_require_validate_but_validate_is_strict(self):
        """Compose is permissive (round-trips unknown presets); validate is the
        publish gate. A caller publishing a name should validate first."""
        spec = SceneAssetSpec("s01", "x", 1, "vr180", "ultra")
        # Compose succeeds (permissive):
        assert compose_scene_name(spec) == "s01_x_seg01_vr180_ultra.mp4"
        # But validate rejects the unknown preset:
        with pytest.raises(NamingError):
            validate_spec(spec)


# ---------------------------------------------------------------------------
# sidecar_scene_fields
# ---------------------------------------------------------------------------


class TestSidecarFields:
    def test_returns_authoritative_fields(self):
        spec = SceneAssetSpec("s03", "santorini", 1, "vr180", "standalone")
        fields = sidecar_scene_fields(spec)
        assert fields == {
            "scene_id": "s03",
            "scene_name": "santorini",
            "segment_index": 1,
            "route": "vr180",
            "preset": "standalone",
            "filename": "s03_santorini_seg01_vr180_standalone.mp4",
        }

    def test_segment_index_is_int_not_string(self):
        """The sidecar stores segment_index as a raw int — the filename may be
        renamed, the JSON is the authoritative source (issue #81 note)."""
        fields = sidecar_scene_fields(SceneAssetSpec("s03", "santorini", 7, "vr180", "standalone"))
        assert isinstance(fields["segment_index"], int)
        assert fields["segment_index"] == 7

    def test_filename_matches_compose(self):
        spec = SceneAssetSpec("s05", "aurora", 2, "fulldome", "pcvr")
        fields = sidecar_scene_fields(spec)
        assert fields["filename"] == compose_scene_name(spec)

    def test_none_fields_resolved_in_sidecar(self):
        fields = sidecar_scene_fields(SceneAssetSpec(route="vr180"))
        assert fields["scene_id"] == "scene"
        assert fields["scene_name"] == "unnamed"
        assert fields["segment_index"] == 1
        assert fields["preset"] == "standalone"

    def test_scene_id_is_the_asset_key_field(self):
        """scene_id is stable across versions — downstream keys on it."""
        fields = sidecar_scene_fields(SceneAssetSpec("s03", "santorini", 1, "vr180", "standalone"))
        # scene_id present and non-empty.
        assert fields["scene_id"] == "s03"

    def test_fields_are_json_serializable(self):
        """The returned dict must be directly json.dumps-able into a sidecar."""
        import json

        fields = sidecar_scene_fields(SceneAssetSpec("s03", "santorini", 1, "vr180", "standalone"))
        # Should not raise.
        json.dumps(fields)
