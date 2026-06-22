"""Inject Spherical Video V2 (sv3d) metadata into MP4 files.

Uses Google's spatial-media CLI for reliable ISOBMFF metadata injection.
Falls back to ffmpeg -movflags remux if spatialmedia is unavailable.

References:
- Google spatial-media: https://github.com/google/spatial-media
- Google Spherical Video V2 spec
"""
import os
import shutil
import subprocess
import tempfile


def inject_spherical_metadata(
    input_path: str,
    output_path: str,
    width: int = 7680,
    height: int = 1920,
    stereo_mode: str = "sbs",
) -> str:
    """Inject Google Spherical Video V2 metadata into an MP4 file.

    Uses Google's spatial-media CLI for correct ISOBMFF sv3d/st3d box injection.
    Falls back to ffmpeg metadata remux if spatialmedia is unavailable.

    Args:
        input_path: Path to input MP4
        output_path: Path to output MP4 with sv3d atom injected
        width: Full panorama width in pixels
        height: Full panorama height in pixels
        stereo_mode: "sbs" (side-by-side) or "tb" (top-bottom)

    Returns:
        Path to output file
    """
    success = _inject_via_spatialmedia_cli(
        input_path, output_path, stereo_mode
    )
    if not success:
        # Fallback: copy file and inject via udta XML
        print("[Metadata] spatial-media not available, using ffmpeg fallback")
        shutil.copy2(input_path, output_path)
        _inject_via_ffmpeg_udta(output_path, stereo_mode)

    return output_path


def _inject_via_spatialmedia_cli(
    input_path: str,
    output_path: str,
    stereo_mode: str,
) -> bool:
    """Inject metadata using Google's spatial-media CLI tool.

    Uses V2 spec (-2 flag) which injects sv3d + st3d ISOBMFF boxes.
    """
    try:
        sm_stereo = "left-right" if stereo_mode == "sbs" else "top-bottom"
        cmd = [
            "python3", "-m", "spatialmedia",
            "-i", "-2",
            "-s", sm_stereo,
            "-p", "equirectangular",
            input_path,
            output_path,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"[Metadata] ✅ VR180 sv3d+st3d injected via spatial-media")
            return True
        else:
            print(f"[Metadata] spatialmedia error: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("[Metadata] python3/spatialmedia not found")
        return False
    except subprocess.TimeoutExpired:
        print("[Metadata] spatialmedia timed out")
        return False
    except Exception as e:
        print(f"[Metadata] spatialmedia error: {e}")
        return False


def _inject_via_ffmpeg_udta(output_path: str, stereo_mode: str):
    """Fallback: inject Spherical Video V1 XML metadata via ffmpeg remux."""
    stereo_tag = "left-right" if stereo_mode == "sbs" else "top-bottom"
    xml = f"""<?xml version="1.0"?>
<rdf:SphericalVideo
 xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">
<GSpherical:Spherical>true</GSpherical:Spherical>
<GSpherical:Stitched>true</GSpherical:Stitched>
<GSpherical:StitchingSoftware>vr180-ai-pipeline</GSpherical:StitchingSoftware>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:StereoMode>{stereo_tag}</GSpherical:StereoMode>
</rdf:SphericalVideo>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        xml_path = f.name

    try:
        tmp = output_path + ".remux.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-c", "copy",
            "-movflags", "+faststart",
            "-metadata:s:v", f"spherical-video={xml}",
            tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            shutil.move(tmp, output_path)
            print("[Metadata] Injected via ffmpeg metadata remux (V1 XML)")
        else:
            print(f"[Metadata] ffmpeg remux failed: {result.stderr[:200]}")
            try:
                os.unlink(tmp)
            except OSError:
                pass
    finally:
        try:
            os.unlink(xml_path)
        except OSError:
            pass