# Stage 4: VR Metadata Embedding

## Goal

Encode the equirectangular stereo frames into an MP4 container with
VR-headset-compatible metadata, producing a playable VR180 video.

## What Gets Embedded

### 1. Spherical Video V2 (RDF/XML)

Google's `<rdf:SphericalVideo>` specification tells VR players the video
is a 180° equirectangular stereo format. The metadata is stored as an
MP4 UUID box (UUID `ffcc8263-f855-4a93-8814-587a02521fdd`).

```xml
<rdf:SphericalVideo ...>
  <GSpherical:Spherical>true</GSpherical:Spherical>
  <GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
  <GSpherical:StereoMode>side-by-side</GSpherical:StereoMode>
  ...
</rdf:SphericalVideo>
```

Without this metadata, VR players treat the video as a flat 2D file.

### 2. Camera Motion Metadata (Optional)

Embed 6-DoF rotation/translation per keyframe as a timed metadata
track. This helps VR headsets smooth head tracking during camera
movement, reducing motion sickness.

- **Rotation**: Quaternion `(w, x, y, z)` per frame
- **Translation**: `(tx, ty, tz)` in metres

### 3. Stereo Mode Flag

The `stereo_mode` metadata tag tells the player to combine the
side-by-side halves into a stereoscopic image.

## Output Specification

| Property | Value |
|----------|-------|
| Container | ISO/IEC 14496-12 (MP4) |
| Video Codec | H.264 (High Profile) / H.265 (Main) |
| Resolution | 7680 × 1920 (SBS) |
| Frame Rate | 60 fps |
| Chroma | 4:2:0 (yuv420p) |
| Stereo Layout | Side-by-side |
| Metadata | Spherical Video V2 XML |
| Playback | Quest, Vision Pro, YouTube VR, VLC |

## Encoding Parameters

| Parameter | H.264 | H.265 |
|-----------|-------|-------|
| Recommended CRF | 23 | 25 |
| Preset | medium | medium |
| Bitrate (approx) | ~50 Mbps | ~25 Mbps |
| Compatibility | All devices | Modern devices only |

## Verification

To verify your output has correct VR metadata:

```bash
# Check spherical XML exists
ffprobe -v quiet -print_format json -show_format -show_streams output.mp4 \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data.get('streams', []):
    print(f'Stream {s[\"index\"]}: {s.get(\"codec_name\")} - '
          f'{s.get(\"width\", \"?\")}x{s.get(\"height\", \"?\")}')
    md = s.get('tags', {})
    if 'spherical-v2' in md:
        print('  ✅ Has Spherical Video V2 metadata')
    if 'stereo_mode' in md:
        print(f'  ✅ Stereo mode: {md[\"stereo_mode\"]}')
"

# Test in VLC
vlc output.mp4

# Upload to YouTube VR with #VR180 tag for headset playback
```

## Known Player Support

| Platform | VR180 Support |
|----------|--------------|
| Meta Quest TV | ✅ Full |
| YouTube VR | ✅ Full |
| Apple Vision Pro | ✅ (Safari + native) |
| VLC | ✅ (with VR plugin) |
| Pico 4 | ✅ |
| WebXR | ✅ (with valid metadata) |