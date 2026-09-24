"""Real storyboard stills through the company LiteLLM gateway (#415).

The gateway exposes the OpenAI-compatible ``POST {base}/v1/images/generations``.
Measured 2026-09-24 (lead):

* ``agnes-image-2.5-flash`` — ~11 s/image, returns ``data[0].url``, no watermark
  (the default);
* ``sensenova-u1-fast`` — ~14 s/image, returns ``data[0].b64_json``, but stamps
  a watermark and adds people unprompted (fallback only).

Both reply shapes are handled.  The HTTP calls are injectable so CI never
touches the network: tests pass fake ``post`` / ``get`` callables.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_IMAGE_MODEL = "agnes-image-2.5-flash"
DEFAULT_VARIANTS = 2
DEFAULT_CONCURRENCY = 3
DEFAULT_RETRIES = 2

#: Composition constraints appended to every shot prompt (the wording the lead
#: used for the verified demo in ``video/storyboard_studio_demo``).
_COMPOSITION = (
    "first-person FPV view, wide ~120 degree field of view, rich depth layers "
    "(foreground / midground / background), level horizon"
)

PostFn = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]
GetFn = Callable[[str, float], bytes]


def size_for_aspect(aspect_ratio: str | None, long_side: int = 1024) -> str:
    """OpenAI-style ``WxH`` for a project aspect ratio (``"1:1"``, ``"16:9"`` ...)."""
    try:
        a, b = (float(x) for x in str(aspect_ratio or "1:1").split(":", 1))
    except ValueError:
        a, b = 1.0, 1.0
    if a <= 0 or b <= 0:
        a, b = 1.0, 1.0
    if a >= b:
        w, h = long_side, round(long_side * b / a)
    else:
        w, h = round(long_side * a / b), long_side
    return f"{w - w % 8}x{h - h % 8}"


def build_still_prompt(shot: dict[str, Any], summary: dict[str, Any] | None = None) -> str:
    """Shot description + theme + style + composition + ``Avoid:`` negatives."""
    summary = summary or {}
    head = str(shot.get("prompt") or shot.get("description") or "").strip()
    parts = [head]
    for key in ("theme", "style"):
        val = str(summary.get(key) or "").strip()
        if val and val not in head:
            parts.append(val)
    parts.append(_COMPOSITION)
    aspect = str(shot.get("aspect_ratio") or summary.get("aspect_ratio") or "").strip()
    if aspect:
        parts.append(f"{aspect} composition")
    text = ", ".join(p for p in parts if p) + "."
    negative = str(summary.get("negative") or "").strip()
    if negative and "avoid:" not in text.lower():
        text += f" Avoid: {negative}, people, vehicles."
    return text


def _httpx_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    import httpx

    resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _httpx_get(url: str, timeout: float) -> bytes:
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


@dataclass
class ImageJob:
    key: str
    prompt: str
    out_path: Path


@dataclass
class ImageResult:
    key: str
    path: str | None
    error: str | None
    attempts: int


class GatewayImageClient:
    """Minimal OpenAI-images client for the LiteLLM gateway."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        model: str = DEFAULT_IMAGE_MODEL,
        size: str = "1024x1024",
        retries: int = DEFAULT_RETRIES,
        timeout: float = 180.0,
        post: PostFn | None = None,
        get: GetFn | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url:
            raise ValueError("gateway image provider needs litellm_base_url (Studio 设置 → Base URL)")
        if not api_key:
            raise ValueError("gateway image provider needs litellm_api_key (Studio 设置 → API Key)")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model or DEFAULT_IMAGE_MODEL
        self.size = size
        self.retries = max(0, int(retries))
        self.timeout = timeout
        self._post = post or _httpx_post
        self._get = get or _httpx_get
        self._sleep = sleep

    def endpoint(self) -> str:
        suffix = "/images/generations" if self.base_url.endswith("/v1") else "/v1/images/generations"
        return self.base_url + suffix

    def generate(self, job: ImageJob) -> ImageResult:
        body = {"model": self.model, "prompt": job.prompt, "n": 1, "size": self.size}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        last_error = "unknown error"
        for attempt in range(1, self.retries + 2):
            try:
                reply = self._post(self.endpoint(), body, headers, self.timeout)
                item = (reply.get("data") or [{}])[0]
                if item.get("b64_json"):
                    data = base64.b64decode(item["b64_json"])
                elif item.get("url"):
                    data = self._get(item["url"], self.timeout)
                else:
                    raise ValueError(f"gateway reply has no image: {str(reply)[:160]}")
                job.out_path.parent.mkdir(parents=True, exist_ok=True)
                job.out_path.write_bytes(data)
                return ImageResult(job.key, str(job.out_path), None, attempt)
            except Exception as exc:  # network, HTTP and decode errors are all retried
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                if attempt <= self.retries:
                    self._sleep(2.0 * attempt)
        return ImageResult(job.key, None, last_error, self.retries + 1)

    def generate_many(self, jobs: list[ImageJob], concurrency: int = DEFAULT_CONCURRENCY) -> list[ImageResult]:
        """Run *jobs* with at most *concurrency* in flight; results keep job order."""
        if not jobs:
            return []
        workers = max(1, min(int(concurrency), len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.generate, jobs))
