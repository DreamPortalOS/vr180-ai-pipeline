"""SeedVR2 video upscaling via ComfyUI backend.

Provides a SeedVR2Upscaler that wraps ComfyUI HTTP API calls for
temporal video super-resolution. CUDA-only; raises clear errors on
CPU/Mac builds. batch_size must be 4n+1 as required by SeedVR2.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BATCH_SIZE_HELP = "batch_size must be 4n+1 (1, 5, 9, 13, 17, ...)"


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if (batch_size - 1) % 4 != 0:
        raise ValueError(f"{_BATCH_SIZE_HELP}, got {batch_size}")


def _assert_cuda() -> None:
    """Raise RuntimeError if CUDA is not available."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed. SeedVR2 requires PyTorch with CUDA.") from None
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available — cannot run SeedVR2Upscaler. "
            "This upscaler requires an NVIDIA GPU with CUDA support. "
            "On Windows, install ComfyUI + SeedVR2 node (see docs/SEEDVR2_SETUP.md)."
        )


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class UpscaleBackend(ABC):
    """Pluggable backend for the actual SeedVR2 upscaling call."""

    @abstractmethod
    def upscale(
        self,
        input_path: str,
        output_path: str,
        factor: int,
        batch_size: int,
    ) -> str:
        """Run upscaling and return the output path."""
        ...


# ---------------------------------------------------------------------------
# ComfyUI backend
# ---------------------------------------------------------------------------


class ComfyUIBackend(UpscaleBackend):
    """Backend that submits a SeedVR2 workflow to the ComfyUI HTTP API.

    Expects ComfyUI to be running at *base_url* (default
    http://127.0.0.1:8188) with:
      - ByteDance-Seed/SeedVR2-3B model downloaded
      - numz/ComfyUI-SeedVR2_VideoUpscaler custom node installed

    Raises a clear actionable error if the server can't be reached or if the
    workflow graph fails.
    """

    # Default workflow template. Keys to fill: INPUT_VIDEO, OUTPUT_VIDEO,
    # FACTOR, BATCH_SIZE.
    # This is a minimal SeedVR2 Video Upscaler workflow JSON.
    _WORKFLOW_TEMPLATE: ClassVar[dict[str, Any]] = {
        "3": {
            "class_type": "LoadVideo",
            "inputs": {
                "video": "INPUT_VIDEO",
                "force_rate": 0,
                "force_size": "Disabled",
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "6": {
            "class_type": "SeedVR2_Upscaler",
            "inputs": {
                "model_name": "seedvr2_ema_3b_fp16.safetensors",
                "scale": "FACTOR",
                "batch_size": "BATCH_SIZE",
                "block_swap": 12,
                "tile_size": 256,
                "overlap": 32,
                "video": ["3", 0],
            },
        },
        "9": {
            "class_type": "VhsSaveVideo",
            "inputs": {
                "frame_rate": 0,
                "loop_count": 0,
                "filename_prefix": "SeedVR2_",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "images": ["6", 0],
            },
        },
    }

    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = self._import_requests()

    # ------------------------------------------------------------------
    # Lazy import of 'requests' so it's not a hard dependency for users
    # that never use the ComfyUI backend.
    # ------------------------------------------------------------------
    @staticmethod
    def _import_requests():
        try:
            import requests as req

            return req
        except ImportError:
            raise ImportError(
                "The 'requests' library is required for ComfyUIBackend. Install it with: pip install requests"
            ) from None

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------
    def _check_connectivity(self) -> None:
        try:
            r = self._session.get(f"{self.base_url}/system_stats", timeout=10)
            r.raise_for_status()
        except self._session.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to ComfyUI at {self.base_url}. "
                "Make sure ComfyUI is running. "
                "See docs/SEEDVR2_SETUP.md for installation instructions."
            ) from None
        except Exception as exc:
            raise RuntimeError(
                f"ComfyUI at {self.base_url} returned an unexpected error: {exc}. See docs/SEEDVR2_SETUP.md."
            ) from exc

    # ------------------------------------------------------------------
    # Workflow submission & polling
    # ------------------------------------------------------------------
    def upscale(
        self,
        input_path: str,
        output_path: str,
        factor: int,
        batch_size: int,
    ) -> str:
        _assert_cuda()
        _validate_batch_size(batch_size)
        self._check_connectivity()

        prompt_id = str(uuid.uuid4())
        workflow = self._build_workflow(
            input_path=input_path,
            factor=factor,
            batch_size=batch_size,
        )

        # Queue the prompt
        payload: dict[str, Any] = {
            "prompt": workflow,
            "client_id": "seedvr2-upscaler",
            "extra_data": {"extra_pnginfo": {}},
        }
        try:
            r = self._session.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            resp = r.json()
        except self._session.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to submit workflow to ComfyUI at {self.base_url}: {exc}") from exc

        # The response may contain a different prompt_id — use the one ComfyUI
        # assigned.
        actual_prompt_id = resp.get("prompt_id", prompt_id)
        log.info(
            "SeedVR2 workflow submitted (prompt_id=%s), polling...",
            actual_prompt_id,
        )

        # Poll for completion (max 600s = 10 min)
        output_filename: str | None = None
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            time.sleep(5)
            try:
                r = self._session.get(
                    f"{self.base_url}/history/{actual_prompt_id}",
                    timeout=10,
                )
                if r.status_code == 404:
                    continue  # not done yet
                r.raise_for_status()
                history = r.json()
            except self._session.exceptions.RequestException:
                continue

            if actual_prompt_id not in history:
                continue

            prompt_data = history[actual_prompt_id]
            status = prompt_data.get("status", {})
            if status.get("completed", False):
                # Grab the first output filename from the workflow outputs
                outputs = prompt_data.get("outputs", {})
                for _node_id, node_out in outputs.items():
                    for out_type in ("gifs", "videos", "images"):
                        items = node_out.get(out_type, [])
                        for item in items:
                            fn = item.get("filename")
                            if fn:
                                output_filename = fn
                                log.info("Output file: %s", fn)
                                break
                    if output_filename:
                        break
                break

            if status.get("error") or status.get("failed"):
                error_detail = prompt_data.get("status", {}).get("error", "unknown")
                raise RuntimeError(f"SeedVR2 workflow failed: {error_detail}. Check the ComfyUI console for details.")

        if not output_filename:
            raise RuntimeError(
                "SeedVR2 workflow did not complete within 10 minutes. Check ComfyUI console for progress."
            )

        # Download the output video
        try:
            r = self._session.get(
                f"{self.base_url}/view",
                params={"filename": output_filename, "type": "output"},
                timeout=300,
            )
            r.raise_for_status()
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(r.content)
            log.info("Downloaded upscaled video to %s", output_path)
        except self._session.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to download output video '{output_filename}': {exc}") from exc

        return output_path

    # ------------------------------------------------------------------
    # Workflow builder
    # ------------------------------------------------------------------
    def _build_workflow(
        self,
        input_path: str,
        factor: int,
        batch_size: int,
    ) -> dict[str, Any]:
        """Fill in the workflow template with concrete values."""
        # Deep-copy the template
        import copy

        wf = copy.deepcopy(self._WORKFLOW_TEMPLATE)

        # Resolve the input video to an absolute path
        abs_input = str(Path(input_path).resolve())

        # Set the node inputs. Our template uses placeholder strings that
        # get replaced here. In a real workflow ComfyUI expects node input
        # slots, but since the node UI may differ, we keep the template
        # adjustable.
        wf["3"]["inputs"]["video"] = abs_input
        wf["6"]["inputs"]["scale"] = factor
        wf["6"]["inputs"]["batch_size"] = batch_size

        return wf


# ---------------------------------------------------------------------------
# SeedVR2 Upscaler (front-facing class)
# ---------------------------------------------------------------------------


class SeedVR2Upscaler:
    """Wrapper for SeedVR2 temporal video super-resolution.

    CUDA-only.  *batch_size* must be **4n+1** (1, 5, 9, 13, …).

    The *backend* argument allows injecting a different backend (e.g.
    for testing or future Path-B headless integration).  Defaults to
    :class:`ComfyUIBackend`.
    """

    def __init__(
        self,
        batch_size: int = 5,
        backend: UpscaleBackend | None = None,
        base_url: str = "http://127.0.0.1:8188",
    ) -> None:
        _assert_cuda()
        _validate_batch_size(batch_size)

        self.batch_size = batch_size
        self.backend = backend or ComfyUIBackend(base_url=base_url)

    def upscale(
        self,
        input_path: str,
        output_path: str,
        factor: int = 2,
    ) -> str:
        """Upscale *input_path* and save to *output_path*.

        Returns the *output_path* on success.
        """
        if factor not in (2, 3, 4):
            raise ValueError(f"factor must be 2, 3, or 4, got {factor}")
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        log.info(
            "SeedVR2Upscaler: %s ×%d (batch_size=%d) → %s",
            input_path,
            factor,
            self.batch_size,
            output_path,
        )

        return self.backend.upscale(
            input_path=input_path,
            output_path=output_path,
            factor=factor,
            batch_size=self.batch_size,
        )
