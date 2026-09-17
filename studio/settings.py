"""Studio runtime settings — provider credentials live here, never in project JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_FILENAME = "studio_settings.json"


@dataclass(frozen=True)
class StudioSettings:
    """Resolved configuration for LLM / generation providers.

    Precedence: explicit constructor args → env vars → optional settings file.
    Secrets must never be written into ``*.studio.json`` project documents.
    """

    litellm_base_url: str = ""
    litellm_api_key: str = ""
    litellm_model: str = ""
    seedance_provider: str = "mock"
    ark_api_key: str = ""
    sensenova_base_url: str = ""
    sensenova_api_key: str = ""
    sensenova_model: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> StudioSettings:
        data: dict[str, Any] = {}
        if path:
            p = Path(path)
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
        elif Path(DEFAULT_SETTINGS_FILENAME).is_file():
            data = json.loads(Path(DEFAULT_SETTINGS_FILENAME).read_text(encoding="utf-8"))

        def pick(key: str, env: str, default: str = "") -> str:
            env_val = os.environ.get(env, "")
            if env_val:
                return env_val
            val = data.get(key, default)
            return str(val) if val is not None else default

        return cls(
            litellm_base_url=pick("litellm_base_url", "STUDIO_LITELLM_BASE_URL"),
            litellm_api_key=pick("litellm_api_key", "STUDIO_LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
            litellm_model=pick("litellm_model", "STUDIO_LITELLM_MODEL", "gpt-4o-mini"),
            seedance_provider=pick("seedance_provider", "STUDIO_SEEDANCE_PROVIDER", "mock"),
            ark_api_key=pick("ark_api_key", "ARK_API_KEY"),
            sensenova_base_url=pick("sensenova_base_url", "STUDIO_SENSENOVA_BASE_URL"),
            sensenova_api_key=pick("sensenova_api_key", "STUDIO_SENSENOVA_API_KEY"),
            sensenova_model=pick("sensenova_model", "STUDIO_SENSENOVA_MODEL", ""),
        )


def settings_from_params(params: dict[str, Any]) -> StudioSettings:
    """Overlay node params onto loaded settings (params win when non-empty)."""
    base = StudioSettings.load()
    overrides = {
        field: str(params[field])
        for field in (
            "litellm_base_url",
            "litellm_api_key",
            "litellm_model",
            "seedance_provider",
            "ark_api_key",
            "sensenova_base_url",
            "sensenova_api_key",
            "sensenova_model",
        )
        if params.get(field)
    }
    if not overrides:
        return base
    payload = {
        "litellm_base_url": base.litellm_base_url,
        "litellm_api_key": base.litellm_api_key,
        "litellm_model": base.litellm_model,
        "seedance_provider": base.seedance_provider,
        "ark_api_key": base.ark_api_key,
        "sensenova_base_url": base.sensenova_base_url,
        "sensenova_api_key": base.sensenova_api_key,
        "sensenova_model": base.sensenova_model,
    }
    payload.update(overrides)
    return StudioSettings(**payload)
