"""LLM polish node — rewrite prompts via LiteLLM-compatible chat API or local mock."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from studio.nodes.base import PortSpec, StudioNode
from studio.settings import settings_from_params

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a VR180 / fulldome prompt editor. Rewrite the user prompt so it is "
    "suitable for immersive video generation: one continuous take, slow steady camera, "
    "clear depth layering, wide FOV, full-frame focus (no shallow DOF), no cuts, "
    "no rapid motion. Keep the core scene and language of the original. "
    "Return only the rewritten prompt, no preamble."
)


class LlmPolishNode(StudioNode):
    type_name = "text.llm_polish"
    category = "script"
    label = "LLM 润色"
    inputs: tuple[PortSpec, ...] = (
        PortSpec(name="prompt", type="text", required=True),
        PortSpec(name="instruction", type="text"),
    )
    outputs: tuple[PortSpec, ...] = (
        PortSpec(name="prompt", type="text"),
        PortSpec(name="meta", type="json"),
    )

    @classmethod
    def param_schema(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "provider",
                "type": "string",
                "default": "mock",
                "label": "provider (mock|litellm|sensenova)",
            },
            {"name": "model", "type": "string", "default": "", "label": "模型名"},
            {"name": "base_url", "type": "string", "default": "", "label": "API Base URL"},
            {"name": "api_key", "type": "string", "default": "", "label": "API Key（也可走环境变量）"},
            {"name": "instruction", "type": "string", "default": "", "label": "额外指令"},
            {"name": "target", "type": "string", "default": "vr180", "label": "目标 (vr180|dome)"},
        ]

    def run(
        self,
        *,
        params: dict[str, Any],
        inputs: dict[str, Any],
        work_dir: str,
        node_id: str,
    ) -> dict[str, Any]:
        _ = work_dir, node_id
        prompt = str(inputs.get("prompt") or params.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("llm_polish requires a non-empty prompt input")
        instruction = str(inputs.get("instruction") or params.get("instruction") or "").strip()
        provider = str(params.get("provider") or "mock").lower()
        target = str(params.get("target") or "vr180")
        settings = settings_from_params(
            {
                "litellm_base_url": params.get("base_url"),
                "litellm_api_key": params.get("api_key"),
                "litellm_model": params.get("model"),
                "sensenova_base_url": params.get("base_url") if provider == "sensenova" else "",
                "sensenova_api_key": params.get("api_key") if provider == "sensenova" else "",
                "sensenova_model": params.get("model") if provider == "sensenova" else "",
            }
        )

        if provider == "mock":
            polished = self._local_polish(prompt, target=target, instruction=instruction)
            return {
                "prompt": polished,
                "meta": {"provider": "mock", "model": "local-rule", "source_chars": len(prompt)},
            }

        if provider == "litellm":
            base_url = settings.litellm_base_url.rstrip("/")
            api_key = settings.litellm_api_key
            model = str(params.get("model") or settings.litellm_model)
            if not base_url:
                raise ValueError("litellm provider requires base_url (param or STUDIO_LITELLM_BASE_URL)")
            if not api_key:
                raise ValueError("litellm provider requires api_key (param, STUDIO_LITELLM_API_KEY, or OPENAI_API_KEY)")
            polished = self._chat(base_url, api_key, model, prompt, instruction)
            return {
                "prompt": polished,
                "meta": {"provider": "litellm", "model": model, "base_url": base_url},
            }

        if provider == "sensenova":
            # SenseNova U1: OpenAI-compatible endpoint if the account exposes one.
            # Free-tier feasibility is tracked in PRD §8.1 / issue #378.
            base_url = settings.sensenova_base_url.rstrip("/")
            api_key = settings.sensenova_api_key
            model = str(params.get("model") or settings.sensenova_model)
            if not base_url or not api_key or not model:
                raise ValueError(
                    "sensenova provider requires base_url, api_key, and model "
                    "(STUDIO_SENSENOVA_* or node params). See PRD §8.1."
                )
            polished = self._chat(base_url, api_key, model, prompt, instruction)
            return {
                "prompt": polished,
                "meta": {"provider": "sensenova", "model": model, "base_url": base_url},
            }

        raise ValueError(f"unknown llm provider {provider!r}; use mock|litellm|sensenova")

    @staticmethod
    def _local_polish(prompt: str, *, target: str, instruction: str) -> str:
        """Offline deterministic rewrite — no network, suitable for CI."""
        from pipeline.prompt_builder import wrap_prompt

        scene = "fpv" if "fly" in prompt.lower() or "fpv" in prompt.lower() else "walkthrough"
        target_key = target if target in {"vr180_flight", "fulldome_180", "vr360_dome"} else "vr180_flight"
        wrapped = wrap_prompt(prompt, scene_type=scene, target=target_key)
        text = str(wrapped.get("positive") or prompt).strip()
        if instruction:
            text = f"{text}\n[editor note: {instruction}]"
        return text

    @staticmethod
    def _chat(base_url: str, api_key: str, model: str, prompt: str, instruction: str) -> str:
        user = prompt if not instruction else f"{prompt}\n\nEditor instruction: {instruction}"
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }
        try:
            res = httpx.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        if res.status_code >= 400:
            detail = res.text[:400]
            raise RuntimeError(f"LLM HTTP {res.status_code}: {detail}")
        try:
            data = res.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"unexpected LLM response shape: {res.text[:300]}") from exc
        text = str(content).strip()
        if not text:
            raise RuntimeError("LLM returned empty content")
        return text
