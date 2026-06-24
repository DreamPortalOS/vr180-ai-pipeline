"""Video generation provider integrations for VR180 Studio.

Provides a unified abstraction over external video generation APIs:
- Kling (快手可灵)
- Seedance (Google-backed)
- Veo (Google DeepMind)

Each provider implements the ``VideoGenProvider`` ABC defined in ``base.py``,
following a three-phase lifecycle: submit → poll → download.
"""
