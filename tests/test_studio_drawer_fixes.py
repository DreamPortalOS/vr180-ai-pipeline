"""#418 lead browser-QA fixes: provider=auto for stills, gallery de-duplication."""

from __future__ import annotations

import pytest

import studio.settings as settings_mod
from studio.graph import extract_gallery
from studio.nodes.production import resolve_stills_provider
from studio.templates import production_pipeline_project


class _S:
    def __init__(self, base: str, key: str) -> None:
        self.litellm_base_url = base
        self.litellm_api_key = key


@pytest.mark.parametrize("explicit", ["mock", "gateway", "file"])
def test_explicit_provider_passes_through(explicit: str) -> None:
    assert resolve_stills_provider({"provider": explicit}) == explicit


def test_auto_is_mock_under_the_test_suite() -> None:
    # conftest sets STUDIO_OFFLINE so no test can reach the real gateway.
    assert resolve_stills_provider({"provider": "auto"}) == "mock"


def test_auto_uses_gateway_when_configured(monkeypatch) -> None:
    monkeypatch.delenv("STUDIO_OFFLINE", raising=False)
    monkeypatch.setattr(settings_mod, "settings_from_params", lambda params: _S("http://gw", "sk"))
    assert resolve_stills_provider({"provider": "auto"}) == "gateway"


def test_auto_falls_back_to_mock_without_settings(monkeypatch) -> None:
    monkeypatch.delenv("STUDIO_OFFLINE", raising=False)
    monkeypatch.setattr(settings_mod, "settings_from_params", lambda params: _S("", ""))
    assert resolve_stills_provider({"provider": "auto"}) == "mock"


def test_production_template_defaults_to_auto() -> None:
    stills = next(n for n in production_pipeline_project().nodes if n.type == "image.batch_stills")
    assert stills.params["provider"] == "auto"


def test_gallery_dedupes_pass_through_nodes() -> None:
    shots = [{"id": f"shot_0{i}", "image": f"/s{i}.png", "description": f"d{i}"} for i in (1, 2, 3)]
    report = {
        "results": {
            "n_stills": {"status": "ok", "outputs": {"stills": {"shots": shots}, "sheet": "/sheet.png"}},
            "n_review": {"status": "ok", "outputs": {"stills": {"shots": shots}}},
        }
    }
    gallery = extract_gallery(report)
    assert [s["id"] for s in gallery["shots"]] == ["shot_01", "shot_02", "shot_03"]
    assert gallery["count"] == 3
    assert all(s["source_node"] == "n_stills" for s in gallery["shots"])


SHOTS = [{"id": "shot_01", "index": 0, "description": "rim"}, {"id": "shot_02", "index": 1, "description": "gorge"}]


def _all_failed(shots, summary, params, out_dir):
    return [{**s, "image": None, "error": "RemoteProtocolError"} for s in shots]


def test_auto_degrades_to_failed_cards_when_gateway_is_down(monkeypatch, tmp_path) -> None:
    import studio.nodes.production as prod

    monkeypatch.setattr(prod, "resolve_stills_provider", lambda params: "gateway")
    monkeypatch.setattr(prod.BatchStillNode, "_gateway_stills", staticmethod(_all_failed))
    out = prod.BatchStillNode().run(
        params={"provider": "auto"},
        inputs={"shots": {"shots": SHOTS, "summary": {}}},
        work_dir=str(tmp_path),
        node_id="n",
    )
    shots = out["stills"]["shots"]
    assert all(s["placeholder"] and s["image"].endswith("_failed.png") for s in shots)
    assert all(s["error"] == "RemoteProtocolError" for s in shots)


def test_explicit_gateway_still_fails_loudly(monkeypatch, tmp_path) -> None:
    import studio.nodes.production as prod

    monkeypatch.setattr(prod.BatchStillNode, "_gateway_stills", staticmethod(_all_failed))
    with pytest.raises(RuntimeError, match="every shot failed"):
        prod.BatchStillNode().run(
            params={"provider": "gateway"},
            inputs={"shots": {"shots": SHOTS, "summary": {}}},
            work_dir=str(tmp_path),
            node_id="n",
        )


def test_gallery_keeps_failure_markers() -> None:
    shots = [
        {"id": "shot_01", "image": "/ok.png"},
        {"id": "shot_02", "image": "/shot_02_failed.png", "placeholder": True, "error": "RemoteProtocolError"},
    ]
    gallery = extract_gallery({"results": {"n": {"status": "ok", "outputs": {"stills": {"shots": shots}}}}})
    ok, bad = gallery["shots"]
    assert ok["placeholder"] is False and ok["error"] is None
    assert bad["placeholder"] is True and bad["error"] == "RemoteProtocolError"
