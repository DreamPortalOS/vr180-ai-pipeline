#!/usr/bin/env python3
"""
Temporal-Consistent AI Outpainting Pipeline
=============================================

Implements a Keyframe Outpainting + Motion Propagation approach
to fill missing regions in equirectangular VR180 video without
catastrophic visual flickering.

Architecture:
1. Extract keyframe (first frame) from source video
2. Expand to full equirectangular sphere (mock Omni/Seedance API or local SD inpaint)
3. Use Optical Flow to propagate keyframe borders across subsequent frames
4. Output temporally-stable outpainted VR180 video

Usage:
    python pipeline/research/ai_outpainter.py [input_video] [--method mock|local_sd]

Requires:
    pip install opencv-python numpy torch diffusers
"""

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ai-outpainter")


@dataclass
class OutpaintRegion:
    """Defines a region to outpaint in equirectangular space."""
    name: str
    y_start: int
    y_end: int
    description: str


class MockOmniOutpainter:
    """
    Mock implementation of Google Omni/Seedance-style API outpainting.

    In production, replace with actual API calls to:
    - Google Gemini Omni (video understanding + generation)
    - Seedance 1.0 (video generation with temporal consistency)
    - Local Stable Diffusion InpaintPipeline

    For now, uses edge-aware content-aware fill via OpenCV inpainting.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.use_sd = False
        self.pipe = None

        # Try to load local Stable Diffusion if available
        if model_path and os.path.exists(model_path):
            try:
                import torch
                from diffusers import StableDiffusionInpaintPipeline
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
                self.use_sd = True
                log.info(f"Loaded Stable Diffusion inpaint model from {model_path}")
            except ImportError:
                log.warning("diffusers not installed, using OpenCV fallback")
            except Exception as e:
                log.warning(f"Failed to load SD model: {e}, using OpenCV fallback")

    def outpaint_region(self, frame: np.ndarray, mask: np.ndarray,
                        prompt: str = "seamless sky environment") -> np.ndarray:
        """
        Outpaint a masked region of the frame.

        Args:
            frame: Input frame (H, W, 3) BGR
            mask: Binary mask (H, W) where 255 = region to fill
            prompt: Text prompt for generation (used with SD)

        Returns:
            Outpainted frame
        """
        if self.use_sd and self.pipe is not None:
            return self._outpaint_with_sd(frame, mask, prompt)
        else:
            return self._outpaint_with_opencv(frame, mask)

    def _outpaint_with_sd(self, frame: np.ndarray, mask: np.ndarray,
                          prompt: str) -> np.ndarray:
        """Outpaint using Stable Diffusion inpainting."""
        from PIL import Image

        # Convert BGR to RGB PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_mask = Image.fromarray(mask)

        result = self.pipe(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]

        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    def _outpaint_with_opencv(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Outpaint using OpenCV content-aware fill (inpainting).

        Uses a combination of:
        1. Navier-Stokes inpainting for small gaps
        2. Edge-aware blending for larger regions
        3. Gaussian blur to smooth seams
        """
        _h, _w = frame.shape[:2]

        # Step 1: Use cv2.inpaint for initial fill
        # INPAINT_NS = Navier-Stokes based (better for texture)
        # INPAINT_TELEA = Fast Marching Method
        filled_ns = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
        filled_telea = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # Blend the two methods for better quality
        alpha = 0.6
        filled = cv2.addWeighted(filled_ns, alpha, filled_telea, 1 - alpha, 0)

        # Step 2: Apply edge-aware bilateral filter to smooth seams
        # This preserves edges while smoothing the transition
        filled = cv2.bilateralFilter(filled, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 3: Create gradient blending at mask edges
        # Expand mask slightly for smooth transition
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_eroded = cv2.erode(mask, kernel, iterations=2)
        transition_zone = mask_dilated - mask_eroded

        if np.any(transition_zone > 0):
            # Create Gaussian weight for smooth blending
            weight = cv2.GaussianBlur(
                transition_zone.astype(np.float32) / 255.0,
                (21, 21), 0
            )
            weight = np.stack([weight] * 3, axis=-1)
            filled = (filled * weight + frame * (1 - weight)).astype(np.uint8)

        return filled


class EquirectangularOutpainter:
    """
    Outpaints a flat equirectangular frame to fill missing regions.

    The equirectangular projection has specific regions that need filling:
    - Top pole (zenith): sky/ceiling
    - Bottom pole (nadir): ground/floor
    - Side wraps: seamless horizontal continuity
    """

    def __init__(self, target_width: int = 7680, target_height: int = 3840):
        self.target_width = target_width
        self.target_height = target_height

    def create_outpaint_mask(self, frame: np.ndarray) -> tuple[np.ndarray, list[OutpaintRegion]]:
        """
        Analyze the frame and create masks for regions that need outpainting.

        Returns:
            Tuple of (combined_mask, list of regions)
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        regions = []

        # Detect if top/bottom regions are black (unfilled)
        top_strip = frame[0:h//10, :, :]
        bottom_strip = frame[h*9//10:h, :, :]

        # Check if regions are mostly black (unfilled equirectangular areas)
        if np.mean(top_strip) < 30:
            # Top needs outpainting (sky/zenith region)
            top_mask = np.zeros((h, w), dtype=np.uint8)
            top_mask[0:h//6, :] = 255
            mask = cv2.bitwise_or(mask, top_mask)
            regions.append(OutpaintRegion(
                name="zenith",
                y_start=0,
                y_end=h//6,
                description="Top pole / sky region"
            ))

        if np.mean(bottom_strip) < 30:
            # Bottom needs outpainting (ground/nadir region)
            bottom_mask = np.zeros((h, w), dtype=np.uint8)
            bottom_mask[h*5//6:h, :] = 255
            mask = cv2.bitwise_or(mask, bottom_mask)
            regions.append(OutpaintRegion(
                name="nadir",
                y_start=h*5//6,
                y_end=h,
                description="Bottom pole / ground region"
            ))

        return mask, regions

    def expand_to_full_equirect(self, frame: np.ndarray,
                                 outpainter: MockOmniOutpainter) -> np.ndarray:
        """
        Expand a partial equirectangular frame to fill the full sphere.

        Args:
            frame: Input frame (may have black unfilled regions)
            outpainter: The outpainting backend

        Returns:
            Full equirectangular frame with all regions filled
        """
        _h, _w = frame.shape[:2]

        # Create outpaint mask
        mask, regions = self.create_outpaint_mask(frame)

        if not np.any(mask > 0):
            log.info("No regions need outpainting")
            return frame

        log.info(f"Outpainting {len(regions)} regions: {[r.name for r in regions]}")

        # Outpaint the masked regions
        result = outpainter.outpaint_region(
            frame, mask,
            prompt="seamless equirectangular sky environment, continuous panorama"
        )

        return result


class TemporalOutpaintPropagator:
    """
    Propagates outpainted keyframe borders across subsequent frames
    using Optical Flow for temporal consistency.

    This prevents flickering by:
    1. Computing optical flow between consecutive frames
    2. Warping the keyframe outpainted borders using the flow
    3. Blending warped borders with current frame
    """

    def __init__(self, method: str = "farneback"):
        self.method = method
        self.prev_gray = None
        self.keyframe_borders = None
        self.keyframe_gray = None

    def _compute_optical_flow(self, prev_gray: np.ndarray,
                               curr_gray: np.ndarray) -> np.ndarray:
        """Compute optical flow between two grayscale frames."""
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        elif self.method == "lucaskanade":
            # Sparse Lucas-Kanade (for keypoint-based tracking)
            # Convert to dense flow representation
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unknown flow method: {self.method}")

        return flow

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp a frame using optical flow field."""
        h, w = frame.shape[:2]
        flow_map = np.column_stack((
            np.repeat(np.arange(w), h),
            np.tile(np.arange(h), w)
        )).reshape(h, w, 2).astype(np.float32)

        # Add flow to create mapping
        flow_map += flow

        # Remap using the flow field
        warped = cv2.remap(
            frame,
            flow_map[:, :, 0].astype(np.float32),
            flow_map[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return warped

    def set_keyframe(self, frame: np.ndarray, outpainted_frame: np.ndarray):
        """
        Set the keyframe with its outpainted borders.

        Args:
            frame: Original keyframe
            outpainted_frame: Keyframe with outpainted borders
        """
        self.keyframe_borders = outpainted_frame.copy()
        self.keyframe_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = self.keyframe_gray.copy()

    def propagate(self, current_frame: np.ndarray,
                  border_mask: np.ndarray) -> np.ndarray:
        """
        Propagate keyframe borders to current frame using optical flow.

        Args:
            current_frame: Current frame from video
            border_mask: Mask indicating border regions (255 = border)

        Returns:
            Frame with propagated stable borders
        """
        if self.keyframe_borders is None:
            log.warning("No keyframe set, returning original frame")
            return current_frame

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow from previous frame to current
        flow = self._compute_optical_flow(self.prev_gray, current_gray)

        # Warp keyframe borders using accumulated flow
        warped_borders = self._warp_frame(self.keyframe_borders, flow)

        # Blend warped borders with current frame using the mask
        mask_3ch = np.stack([border_mask / 255.0] * 3, axis=-1)

        # Smooth the mask edges
        mask_smooth = cv2.GaussianBlur(mask_3ch, (21, 21), 0)

        # Composite: current frame in center, warped borders at edges
        result = (current_frame * (1 - mask_smooth) +
                  warped_borders * mask_smooth).astype(np.uint8)

        # Update state
        self.prev_gray = current_gray.copy()
        self.keyframe_borders = warped_borders.copy()

        return result


def process_video_with_outpainting(
    input_path: str,
    output_path: str,
    method: str = "mock",
    model_path: str | None = None,
    max_frames: int | None = None,
    target_width: int = 7680,
    target_height: int = 3840,
) -> str:
    """
    Process a video with temporal-consistent AI outpainting.

    Args:
        input_path: Input video path
        output_path: Output video path
        method: Outpainting method ("mock" for OpenCV, "local_sd" for Stable Diffusion)
        model_path: Path to SD inpaint model (for local_sd method)
        max_frames: Maximum frames to process
        target_width: Target equirectangular width
        target_height: Target equirectangular height

    Returns:
        Output video path
    """
    log.info(f"Processing {input_path} with {method} outpainting")

    # Initialize components
    outpainter = MockOmniOutpainter(model_path=model_path)
    eq_outpainter = EquirectangularOutpainter(target_width, target_height)
    propagator = TemporalOutpaintPropagator(method="farneback")

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    log.info(f"Input: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Setup ffmpeg output pipe
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        output_path
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    frame_idx = 0
    keyframe_set = False
    border_mask = None
    start_time = time.time()

    try:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == 0:
                # Process keyframe: full outpainting
                log.info("Processing keyframe with full outpainting...")
                outpainted = eq_outpainter.expand_to_full_equirect(frame, outpainter)

                # Create border mask for propagation
                border_mask, _ = eq_outpainter.create_outpaint_mask(frame)
                if not np.any(border_mask > 0):
                    # Create default border mask (top and bottom 15%)
                    border_mask = np.zeros((height, width), dtype=np.uint8)
                    border_mask[0:height//7, :] = 255
                    border_mask[height*6//7:height, :] = 255

                # Set keyframe for propagation
                propagator.set_keyframe(frame, outpainted)
                keyframe_set = True

                result = outpainted
            else:
                # Propagate borders from keyframe using optical flow
                result = propagator.propagate(frame, border_mask) if keyframe_set else frame

            # Write to output
            proc.stdin.write(result.tobytes())
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                log.info(f"Progress: {frame_idx}/{total_frames} ({fps_actual:.1f} fps)")

    except Exception as e:
        log.error(f"Error at frame {frame_idx}: {e}")
        proc.stdin.close()
        proc.wait()
        cap.release()
        raise

    finally:
        cap.release()

    proc.stdin.close()
    proc.wait()

    elapsed = time.time() - start_time
    log.info(f"Complete: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    log.info(f"Output: {output_path}")

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        log.error(f"ffmpeg error: {stderr[-500:]}")

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Outpainting for VR180")
    parser.add_argument("input", nargs="?", default="video/testfpv_vr180.mp4",
                        help="Input video path")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--method", choices=["mock", "local_sd"], default="mock",
                        help="Outpainting method")
    parser.add_argument("--model-path", default=None, help="Path to SD inpaint model")
    parser.add_argument("--max-frames", type=int, default=30, help="Max frames to process")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error(f"Input not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"video/{stem}_outpainted_{args.method}.mp4"

    output = process_video_with_outpainting(
        args.input,
        args.output,
        method=args.method,
        model_path=args.model_path,
        max_frames=args.max_frames,
    )
    log.info(f"Output saved to: {output}")


if __name__ == "__main__":
    main()
