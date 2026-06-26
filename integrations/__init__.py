"""Video generation provider integrations for VR180 AI Pipeline.

Provides a unified interface over external video generation APIs:
- Kling (Kuaishou / 可灵)
- Seedance (Google-backed)
- Veo (Google DeepMind / Vertex AI)

Each provider implements the :class:`VideoGenProvider` ABC defined in
:mod:`integrations.base`, exposing a single synchronous ``generate()`` method.
"""

from __future__ import annotations
