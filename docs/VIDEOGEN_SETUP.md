# Video Generation Providers — Setup Guide

This document describes how to configure and use the three supported external
video generation providers: **Kling** (Kuaishou), **Seedance** (Google-backed),
and **Veo** (Google DeepMind / Vertex AI).

## Quick Start

```bash
# Set your API key(s) — only the one(s) you need
export KLING_API_KEY="your-kling-api-key"
# export SEEDANCE_API_KEY="your-seedance-api-key"
# export VEO_API_KEY="your-veo-api-key"
# export GCP_PROJECT_ID="your-gcp-project"  # only for Veo

# Generate a 5-second video with the default provider (kling)
python -m scripts.generate "fly over mountains"

# Use a specific provider
python -m scripts.generate "walkthrough of a temple" --provider seedance

# Wrap prompt through prompt_builder for VR180 optimisation
python -m scripts.generate "underwater coral reef" --provider veo --target-aware --scene walkthrough

# List available providers
python -m scripts.generate --list-providers
```

## Provider Overview

| Provider   | Env Variable        | Default Model          | Notes                        |
|------------|---------------------|------------------------|------------------------------|
| Kling      | `KLING_API_KEY`     | `kling-v1`             | Submit + poll lifecycle      |
| Seedance   | `SEEDANCE_API_KEY`  | (API default)          | Submit + poll lifecycle      |
| Veo        | `VEO_API_KEY`       | `veo-001`              | Synchronous predict, GCP     |

All three implement the same `VideoGenProvider` ABC with a synchronous
`generate()` call that blocks until the video URL is available.

---

## 1. Kling (Kuaishou / 可灵)

### Obtaining an API Key

1. Go to https://docs.klingai.com/
2. Sign up for a developer account
3. Create an API key from the console
4. Set the key in your environment:

```bash
export KLING_API_KEY="your-kling-api-key"
```

### Usage

```python
from integrations.factory import get_provider

provider = get_provider("kling")
result = provider.generate(
    prompt="a scenic drone flyover of the Grand Canyon",
    duration=5,
    aspect_ratio="16:9",
    fps=24,
    cfg_scale=7.0,
    negative_prompt="blurry, low quality",
)
print(result.video_url)
```

### Supported Parameters

| Parameter        | Type  | Description                             |
|------------------|-------|-----------------------------------------|
| `model`          | str   | Model name (default `kling-v1`)        |
| `cfg_scale`      | float | Classifier-free guidance scale (1-30)   |
| `negative_prompt`| str   | Things to avoid in the generated video  |

---

## 2. Seedance (Google-backed)

### Obtaining an API Key

1. Go to https://docs.seedance.ai/
2. Register for an account
3. Generate an API key from the dashboard
4. Set the environment variable:

```bash
export SEEDANCE_API_KEY="your-seedance-api-key"
```

### Usage

```python
from integrations.factory import get_provider

provider = get_provider("seedance")
result = provider.generate(
    prompt="slow cinematic pan across a misty forest",
    duration=5,
    aspect_ratio="16:9",
    fps=24,
)
print(result.video_url)
```

### Supported Parameters

| Parameter        | Type  | Description                             |
|------------------|-------|-----------------------------------------|
| `model`          | str   | Model ID (optional, uses default)       |
| `negative_prompt`| str   | Things to avoid in the generated video  |

---

## 3. Veo (Google DeepMind / Vertex AI)

### Obtaining an API Key

1. Go to https://console.cloud.google.com/apis/credentials
2. Create a project (or use existing)
3. Enable the **Vertex AI API**
4. Create an API key
5. Set environment variables:

```bash
export VEO_API_KEY="your-vertex-ai-api-key"
export GCP_PROJECT_ID="your-gcp-project-id"  # defaults to "my-project" if unset
```

**Note:** Veo uses a synchronous `predict` endpoint — results are typically
available within 60–180 seconds.

### Usage

```python
from integrations.factory import get_provider

provider = get_provider("veo")
result = provider.generate(
    prompt="aerial view of a futuristic city at sunset",
    duration=5,
    aspect_ratio="16:9",
    fps=24,
    sample_count=1,
)
print(result.video_url)
```

### Supported Parameters

| Parameter        | Type  | Description                             |
|------------------|-------|-----------------------------------------|
| `negative_prompt`| str   | Things to avoid                         |
| `sample_count`   | int   | Number of samples to generate (default 1)|

---

## Programmatic Usage (All Providers)

```python
from integrations.factory import get_provider, list_providers

print(list_providers())  # ["kling", "seedance", "veo"]

# Provider selection by name (case-insensitive)
provider = get_provider("KLING")

# Pass api_key explicitly (overrides env var)
provider = get_provider("seedance", api_key="my-custom-key")

# Generate
result = provider.generate(
    prompt="cinematic shot of a mountain lake",
    duration=5,
    aspect_ratio="16:9",
    fps=30,
)

# Result fields
print(result.video_url)     # downloadable URL
print(result.job_id)        # provider's job identifier (may be None)
print(result.provider)      # "kling", "seedance" or "veo"
print(result.metadata)      # raw API response dict
```

## Architecture

All providers inherit from `integrations.base.VideoGenProvider` (abstract base
class). The factory in `integrations.factory` lazily imports and registers
each provider. Environment variables are read at instantiation time — missing
keys raise `ValueError` with a clear message.

```
scripts/generate.py  (CLI)
       │
       ▼
integrations/factory.py  (get_provider / list_providers)
       │
       ├── integrations/kling.py    (httpx, submit+poll)
       ├── integrations/seedance.py (httpx, submit+poll)
       └── integrations/veo.py      (httpx, sync predict)
```

## Testing

All tests are fully mocked — no real API calls are made:

```bash
pytest tests/test_integrations.py -v
