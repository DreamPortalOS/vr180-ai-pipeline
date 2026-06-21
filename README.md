vr180-ai-pipeline
=================

.. container:: badges

   |License: MIT| |Python 3.10+|

2D AI Video → VR180 Immersive Experience Conversion Pipeline

Convert AI-generated 2D videos (720p, 16:9) into VR180 format for immersive
headset viewing. The pipeline applies **depth estimation**, **stereoscopic
disparity rendering**, **equirectangular (180°) sphere projection**, and **VR
metadata embedding** — producing a side-by-side VR180 video compatible with
Meta Quest, Apple Vision Pro, and other VR headsets.

.. mermaid::

   flowchart TD
       A["2D AI Video (720p, 16:9)"] --> B["Depth Estimation<br/><i>Depth Anything V2 / MiDaS</i>"]
       B --> C["Stereo Disparity<br/><i>Left/Right View Generation</i>"]
       C --> D["Equirectangular Projection<br/><i>3840 × 1920 Sphere Map</i>"]
       D --> E["VR Metadata Embedding<br/><i>Spherical V2 + Camera Motion</i>"]
       E --> F["VR180 Output<br/><i>SBS H.264/H.265</i>"]

---

Pipeline Stages
---------------

**Stage 1 — Depth Estimation**
   Per-frame metric depth estimation using Depth Anything V2 or MiDaS. Each
   frame of the input video is processed to produce a dense depth map.

**Stage 2 — Stereo Disparity**
   Using the depth map + optical flow, a horizontal parallax shift is applied:
   *Left eye* = original image shifted right; *Right eye* = original image
   shifted left. Inpainting fills disoccluded regions.

**Stage 3 — Equirectangular Projection**
   The planar stereo pair is mapped onto a 180° hemisphere via equirectangular
   projection (3840 × 1920 resolution). A grid-mesh transform warps the flat
   image into spherical coordinates that VR headsets can render natively.

**Stage 4 — VR Metadata Embedding**
   The output video is stamped with:
   - *Spherical Video V2* metadata ( `<rdf:SphericalVideo>` in the MP4)
   - *Camera Motion Metadata* for 6-DoF head tracking hints
   Output: side-by-side (SBS) H.264/H.265 MP4 at ~60 fps.

---

Quick Start
-----------

.. code:: bash

   # Install dependencies
   pip install -r requirements.txt

   # Download model weights (one-time)
   python scripts/download_models.py

   # Run the full pipeline
   python scripts/run_pipeline.py --input video.mp4 --output vr180.mp4

   # Individual stages
   python scripts/run_pipeline.py --input video.mp4 --stage depth     --output depth/
   python scripts/run_pipeline.py --input video.mp4 --stage stereo    --output stereo/
   python scripts/run_pipeline.py --input video.mp4 --stage equirect  --output sphere.mp4
   python scripts/run_pipeline.py --input video.mp4 --stage metadata  --output vr180.mp4

---

System Requirements
-------------------

+-------------------+------------------------------------------------------+
| Component         | Minimum                                              |
+===================+======================================================+
| GPU               | NVIDIA RTX 3060 12GB / Apple M2 Max (32GB unified)   |
+-------------------+------------------------------------------------------+
| RAM               | 32 GB                                                |
+-------------------+------------------------------------------------------+
| Storage           | 10 GB (models + temp)                                |
+-------------------+------------------------------------------------------+
| Python            | 3.10+                                                |
+-------------------+------------------------------------------------------+
| Frameworks        | PyTorch 2.x + CUDA 12.x / MPS (Apple Silicon)        |
+-------------------+------------------------------------------------------+

---

Project Structure
-----------------

::

   vr180-ai-pipeline/
   ├── README.md
   ├── LICENSE
   ├── pyproject.toml
   ├── requirements.txt
   ├── .gitignore
   ├── docs/
   │   ├── architecture.md
   │   ├── pipeline-overview.md
   │   ├── depth-estimation.md
   │   ├── stereo-disparity.md
   │   ├── equirectangular-projection.md
   │   └── vr-metadata-embedding.md
   ├── pipeline/
   │   ├── __init__.py
   │   ├── depth_estimator.py
   │   ├── stereo_renderer.py
   │   ├── equirectangular_mapper.py
   │   └── vr_metadata.py
   └── scripts/
       ├── run_pipeline.py
       └── download_models.py

---

Documentation
-------------

Detailed documentation is in the ``docs/`` directory:

- **Architecture**:  ``docs/architecture.md`` — system design, data flow, component interaction
- **Pipeline Overview**:  ``docs/pipeline-overview.md`` — end-to-end walkthrough
- **Stage 1 — Depth**:  ``docs/depth-estimation.md``
- **Stage 2 — Stereo**:  ``docs/stereo-disparity.md``
- **Stage 3 — Equirect**:  ``docs/equirectangular-projection.md``
- **Stage 4 — VR Metadata**:  ``docs/vr-metadata-embedding.md``

---

License
-------

MIT License — see ``LICENSE`` for details.
