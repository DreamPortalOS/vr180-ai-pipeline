"""Tests for the Prompt Library (pipeline/prompt_library.py).

Verifies the acceptance criteria of issue #174:
- ``TEMPLATES`` has ≥ 8 entries; every key is unique; every template's
  declared ``placeholders`` exactly matches the placeholders referenced in
  its ``body`` (parametrized per-template).
- ``render`` raises ``PromptLibraryError`` on missing or extra fields, with
  the offending field name(s) named in the message.
- ``get_template`` raises ``KeyError`` for unknown keys and lists the
  available keys in the message.
- ``list_templates(camera_motion=...)`` returns only matching templates.
- A rendered prompt feeds directly into
  ``pipeline.prompt_builder.wrap_prompt_for_vr180`` without error.
"""

from __future__ import annotations

import re

import pytest

from pipeline.prompt_builder import wrap_prompt_for_vr180
from pipeline.prompt_library import (
    TEMPLATES,
    PromptLibraryError,
    PromptTemplate,
    get_template,
    list_templates,
    render,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholders_in_body(body: str) -> set[str]:
    """Names of ``{name}`` field references actually appearing in *body*."""
    import string

    return {name for _, name, _, _ in string.Formatter().parse(body) if name}


# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------


class TestRegistryInvariants:
    """The TEMPLATES registry as a whole must be well-formed."""

    def test_has_at_least_eight_templates(self):
        assert len(TEMPLATES) >= 8

    def test_keys_are_unique(self):
        assert len(TEMPLATES) == len(set(TEMPLATES))

    def test_all_values_are_prompt_templates(self):
        for value in TEMPLATES.values():
            assert isinstance(value, PromptTemplate)

    def test_required_camera_motions_covered(self):
        """Cover at least the eight motion families named in the spec."""
        required = {
            "static",
            "dolly_in",
            "dolly_out",
            "orbit_left",
            "orbit_right",
            "pan_slow",
            "tilt_up",
            "push_through",
        }
        present = {t.camera_motion for t in TEMPLATES.values()}
        missing = required - present
        assert not missing, f"missing camera motions: {sorted(missing)}"

    def test_list_templates_default_returns_all(self):
        listed = list_templates()
        assert len(listed) == len(TEMPLATES)
        assert {t.key for t in listed} == set(TEMPLATES)

    def test_list_templates_returns_copies_not_registry_alias(self):
        """list_templates() must return independent list objects."""
        a = list_templates()
        b = list_templates()
        assert a is not b
        assert a == b


# ---------------------------------------------------------------------------
# Per-template placeholder consistency (parametrized)
# ---------------------------------------------------------------------------


@pytest.fixture(params=sorted(TEMPLATES))
def template(request: pytest.FixtureRequest) -> PromptTemplate:
    return TEMPLATES[request.param]


class TestPerTemplateConsistency:
    """Every template must be internally consistent."""

    def test_key_matches_registry_key(self, template: PromptTemplate):
        assert TEMPLATES[template.key] is template

    def test_key_is_nonempty_identifier(self, template: PromptTemplate):
        assert template.key
        assert re.fullmatch(r"[a-z0-9_]+", template.key), template.key

    def test_summary_nonempty(self, template: PromptTemplate):
        assert template.summary and template.summary.strip()

    def test_body_nonempty(self, template: PromptTemplate):
        assert template.body and template.body.strip()

    def test_camera_motion_nonempty(self, template: PromptTemplate):
        assert template.camera_motion and template.camera_motion.strip()

    def test_recommended_duration_positive(self, template: PromptTemplate):
        assert template.recommended_duration > 0

    def test_placeholders_match_body(self, template: PromptTemplate):
        """Declared placeholders == placeholders actually referenced in body."""
        actual = _placeholders_in_body(template.body)
        declared = set(template.placeholders)
        assert actual == declared, (
            f"template {template.key!r} mismatch — "
            f"in body not declared: {sorted(actual - declared)}; "
            f"declared not in body: {sorted(declared - actual)}"
        )

    def test_template_is_frozen(self, template: PromptTemplate):
        """frozen=True dataclass must reject attribute mutation."""
        with pytest.raises((AttributeError, Exception)):
            template.key = "mutated"  # type: ignore[misc]

    def test_body_renders_with_declared_placeholders(self, template: PromptTemplate):
        """Supplying every declared placeholder must render without error."""
        fields = {name: f"VAL_{name}" for name in template.placeholders}
        out = render(template.key, **fields)
        assert isinstance(out, str)
        assert out.strip()
        # every supplied value must appear in the output
        for value in fields.values():
            assert value in out

    @pytest.mark.parametrize(
        "constraint",
        [
            "no cuts",
            "constant-speed",
            "stable",
            "foreground",
            "background",
            "avoid large mirrors",
        ],
    )
    def test_body_carries_vr180_constraints(self, template: PromptTemplate, constraint: str):
        """Every template body embeds the VR180-friendly constraint backbone."""
        assert constraint.lower() in template.body.lower(), (
            f"template {template.key!r} body missing constraint phrase {constraint!r}"
        )


# ---------------------------------------------------------------------------
# render() error handling
# ---------------------------------------------------------------------------


class TestRenderErrors:
    """render() must fail loudly on missing / extra fields."""

    def test_missing_field_raises(self):
        with pytest.raises(PromptLibraryError, match="setting") as exc:
            render("slow_dolly_in", subject="a sailboat")
        assert "setting" in str(exc.value)

    def test_missing_all_fields_raises(self):
        with pytest.raises(PromptLibraryError) as exc:
            render("slow_dolly_in")
        msg = str(exc.value)
        assert "subject" in msg
        assert "setting" in msg

    def test_extra_field_raises(self):
        with pytest.raises(PromptLibraryError, match="extra") as exc:
            render(
                "slow_dolly_in",
                subject="a sailboat",
                setting="a harbour",
                extraneous="oops",
            )
        assert "extraneous" in str(exc.value)

    def test_typo_field_raises_rather_than_swallowed(self):
        """A typo'd field name must surface, not silently render an empty."""
        with pytest.raises(PromptLibraryError) as exc:
            render("slow_dolly_in", subject="a sailboat", seting="a harbour")
        assert "setting" in str(exc.value)

    def test_unknown_key_raises_keyerror(self):
        with pytest.raises(KeyError) as exc:
            render("does_not_exist", subject="x", setting="y")
        msg = str(exc.value)
        # message must list the available keys
        for key in TEMPLATES:
            assert key in msg

    def test_render_returns_str(self):
        out = render("locked_static", subject="a vase", setting="a still room")
        assert isinstance(out, str)
        assert "a vase" in out
        assert "a still room" in out


# ---------------------------------------------------------------------------
# get_template()
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_returns_registered_template(self):
        t = get_template("orbit_left")
        assert t is TEMPLATES["orbit_left"]

    def test_unknown_key_raises_keyerror_listing_all(self):
        with pytest.raises(KeyError) as exc:
            get_template("totally_unknown_key")
        msg = str(exc.value)
        assert "totally_unknown_key" in msg
        for key in TEMPLATES:
            assert key in msg, f"available-key {key!r} not mentioned in error"

    def test_keyerror_is_subclass_of_lookuperror(self):
        with pytest.raises(LookupError):
            get_template("nope")


# ---------------------------------------------------------------------------
# list_templates() filtering
# ---------------------------------------------------------------------------


class TestListTemplatesFilter:
    def test_filter_dolly_in(self):
        matched = list_templates(camera_motion="dolly_in")
        assert matched, "expected at least one dolly_in template"
        for t in matched:
            assert t.camera_motion == "dolly_in"
        # and nothing was missed
        expected = {t.key for t in TEMPLATES.values() if t.camera_motion == "dolly_in"}
        assert {t.key for t in matched} == expected

    def test_filter_static(self):
        matched = list_templates(camera_motion="static")
        for t in matched:
            assert t.camera_motion == "static"

    def test_filter_unknown_motion_returns_empty(self):
        assert list_templates(camera_motion="teleport_nowhere") == []

    def test_filter_returns_prompt_template_instances(self):
        for t in list_templates(camera_motion="dolly_in"):
            assert isinstance(t, PromptTemplate)


# ---------------------------------------------------------------------------
# Integration with prompt_builder (call-only, do not modify prompt_builder)
# ---------------------------------------------------------------------------


class TestPromptBuilderIntegration:
    """Rendered text must drop straight into wrap_prompt_for_vr180."""

    @pytest.mark.parametrize("scene_type", ["fpv", "walkthrough", "orbit", "static"])
    def test_rendered_feeds_wrap_prompt_for_vr180(self, scene_type: str):
        text = render("orbit_left", subject="a marble statue", setting="a sunlit plaza")
        result = wrap_prompt_for_vr180(text, scene_type=scene_type)
        assert isinstance(result, dict)
        assert {"positive", "negative"} <= set(result)
        # the template text must survive verbatim as the positive prefix
        assert result["positive"].startswith(text)

    def test_every_template_renders_into_wrap(self):
        """Every template, fully rendered, must be wrap-compatible."""
        for key in TEMPLATES:
            template = TEMPLATES[key]
            fields = {name: f"VAL_{name}" for name in template.placeholders}
            text = render(key, **fields)
            # wrap_prompt_for_vr180 must not raise
            result = wrap_prompt_for_vr180(text, scene_type="static")
            assert text in result["positive"]
