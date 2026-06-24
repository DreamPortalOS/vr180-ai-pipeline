"""Factory for instantiating video generation providers.

Usage::

    from integrations.factory import get_provider

    provider = get_provider("kling")
    job_id = await provider.submit(params)
"""

from __future__ import annotations

import logging
from typing import Any

from integrations.base import VideoGenProvider

log = logging.getLogger(__name__)

# Lazy registry — providers are imported on first access to avoid
# pulling in heavy dependencies (httpx, hashlib, etc.) at import time.
_PROVIDER_REGISTRY: dict[str, type[VideoGenProvider]] = {}


def _ensure_registry() -> None:
    """Populate the provider registry on first call."""
    if _PROVIDER_REGISTRY:
        return

    # Lazy imports to keep factory import lightweight
    from integrations.kling import KlingProvider
    from integrations.seedance import SeedanceProvider
    from integrations.veo import VeoProvider

    _PROVIDER_REGISTRY["kling"] = KlingProvider
    _PROVIDER_REGISTRY["seedance"] = SeedanceProvider
    _PROVIDER_REGISTRY["veo"] = VeoProvider

    log.debug("Provider registry: %s", list(_PROVIDER_REGISTRY))


def get_provider(name: str, **kwargs: Any) -> VideoGenProvider:
    """Return a configured ``VideoGenProvider`` instance by name.

    Parameters
    ----------
    name : str
        One of ``"kling"``, ``"seedance"``, ``"veo"``.
    **kwargs
        Passed through to the provider constructor (e.g. ``api_key``).

    Returns
    -------
    VideoGenProvider
        A fully initialised provider instance.

    Raises
    ------
    ValueError
        If *name* is not a recognised provider.
    """
    _ensure_registry()
    cls = _PROVIDER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown provider {name!r}. Available: {list(_PROVIDER_REGISTRY)}")
    instance = cls(**kwargs)
    log.info("Provider '%s' instantiated (%s)", name, cls.__name__)
    return instance


def list_providers() -> list[str]:
    """Return the list of registered provider names."""
    _ensure_registry()
    return list(_PROVIDER_REGISTRY)
