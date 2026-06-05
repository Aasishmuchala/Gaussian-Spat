#!/usr/bin/env python3
"""
Main 4DGS Pipeline Runner

Orchestrates the complete 4D Gaussian Splatting pipeline:
1. Multi-view 3D reconstruction (preprocessing)
2. Keyframe processing with velocity estimation
3. 4D Gaussian training

Usage:
    python main_pipeline.py --videos "cam1.mp4,cam2.mp4,cam3.mp4" --output-dir ./output
"""

import argparse
import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Callable

# Add lib/pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import MultiViewReconstructor
from keyframe_processor import process_keyframes
from train_4dgs import train_4d_gaussians

_progress_callback: Optional[Callable] = None

def set_progress_callback(callback: Callable):
    global _progress_callback
    _progress_callback = callback

def emit(stage: str, progress: float, message: str, metadata: Dict = None):
    if _progress_callback:
        _progress_callback({
            "stage": stage,
            "progress": progress,
            "message": message,
            "metadata": metadata or {}
        })
    print(f"[{progress:.1f}%] {message}", flush=True)


def run_pipeline(
    video_paths: List[str],
    output_dir: str,
    fps: float = 2.0,
    max_frames: int = 60,
    keyframe_step: int = 5,
    max_training_steps: int = 30000,
    num_points: int = 100000,
    cleanup_temp: bool = True
) -> Dict:
    """
    Run the complete 4D Gaussian Splatting pipeline.

    Args:
        video_paths: List of paths to input videos
        output_dir: Directory for all outputs
        fps: Frame rate for extraction
        max_frames: Maximum frames to process
        keyframe_step: Interval for keyframe selection
        max_training_steps: Maximum training iterations
        num_points: Target number of Gaussians
        cleanup_temp: Whether to cleanup temporary files

    Returns:
        Dict with pipeline results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    emit("setup", 0, f"Starting 4DGS Pipeline with {len(video_paths)} videos...")

    result = {
        "success": False,
        "stages": {},
        "output_dir": str(output_dir)
    }

    try:
        # =============================================
        # STAGE 1: Multi-view 3D Reconstruction
        # =============================================
        emit("running_colmap", 0, "Stage 1: Extracting frames and reconstructing 3D points...")

        reconstructor = MultiViewReconstructor(
            video_paths=video_paths,
            output_dir=str(temp_dir / "reconstruction"),
            fps=fps,
            max_frames=max_frames,
            num_features=2000,
            min_matches=15,
            min_triangulation_angle=1.5
        )

        recon_result = reconstructor.run()

        if not recon_result.get("success"):
            result["error"] = f"Reconstruction failed: {recon_result.get('error', 'Unknown error')}"
            return result

        result["stages"]["reconstruction"] = recon_result
        num_frames = recon_result.get("num_frames", 0)
        total_points = recon_result.get("total_points", 0)

        emit("running_colmap", 100,
            f"Stage 1 complete: {total_points} points from {num_frames} frames")

        # =============================================
        # STAGE 2: Keyframe Processing
        # =============================================
        emit("processing_keyframes", 0, "Stage 2: Processing keyframes with velocity estimation...")

        keyframes_npz = str(temp_dir / "keyframes.npz")

        keyframe_result = process_keyframes(
            input_dir=str(temp_dir / "reconstruction" / "points"),
            output_path=keyframes_npz,
            frame_start=0,
            frame_end=num_frames,
            keyframe_step=keyframe_step,
            target_points=num_points
        )

        if not keyframe_result.get("success"):
            result["error"] = f"Keyframe processing failed: {keyframe_result.get('error', 'Unknown error')}"
            return result

        result["stages"]["keyframes"] = keyframe_result
        num_gaussians = keyframe_result.get("num_points", 0)

        emit("processing_keyframes", 100,
            f"Stage 2 complete: {num_gaussians} Gaussians prepared")

        # =============================================
        # STAGE 3: 4D Gaussian Training
        # =============================================
        emit("training_4dgs", 0, "Stage 3: Training 4D Gaussian splatting...")

        training_result = train_4d_gaussians(
            init_npz_path=keyframes_npz,
            output_dir=str(output_dir / "training"),
            max_steps=max_training_steps,
            image_width=640,
            image_height=480
        )

        if not training_result.get("success"):
            result["error"] = f"Training failed: {training_result.get('error', 'Unknown error')}"
            return result

        result["stages"]["training"] = training_result

        emit("training_4dgs", 100,
            f"Stage 3 complete: {training_result.get('num_gaussians')} Gaussians trained")

        # =============================================
        # STAGE 4: Prepare Viewer Data
        # =============================================
        emit("viewing", 0, "Stage 4: Preparing viewer data...")

        # Copy viewer data to accessible location
        viewer_source = Path(training_result.get("viewer_data_path", ""))
        if viewer_source.exists():
            viewer_dest = output_dir / "viewer_data.npz"
            shutil.copy(viewer_source, viewer_dest)
            result["viewer_data"] = str(viewer_dest)

        emit("viewing", 100, "Stage 4 complete")

        # Cleanup temp directory
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)

        result["success"] = True
        result["num_frames"] = num_frames
        result["num_gaussians"] = training_result.get("num_gaussians", 0)
        result["checkpoint_path"] = training_result.get("checkpoint_path", "")
        result["message"] = f"Pipeline complete! {result['num_gaussians']} 4D Gaussians trained."

        emit("complete", 100, result["message"])

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="4D Gaussian Splatting Pipeline")
    parser.add_argument("--videos", required=True, help="Comma-separated video paths")
    parser.add_argument("--output-dir", default="./4dgs_output", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame rate for extraction")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum frames")
    parser.add_argument("--keyframe-step", type=int, default=5, help="Keyframe interval")
    parser.add_argument("--max-steps", type=int, default=30000, help="Training iterations")
    parser.add_argument("--num-points", type=int, default=100000, help="Target Gaussians")

    args = parser.parse_args()

    video_paths = [v.strip() for v in args.videos.split(",")]

    print("=" * 60)
    print("4D Gaussian Splatting Pipeline")
    print("=" * 60)
    print(f"Input videos: {len(video_paths)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max frames: {args.max_frames}")
    print(f"Keyframe step: {args.keyframe_step}")
    print(f"Training steps: {args.max_steps}")
    print("=" * 60)

    result = run_pipeline(
        video_paths=video_paths,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        keyframe_step=args.keyframe_step,
        max_training_steps=args.max_steps,
        num_points=args.num_points
    )

    if result["success"]:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Frames processed: {result['num_frames']}")
        print(f"Gaussians trained: {result['num_gaussians']}")
        print(f"Output directory: {result['output_dir']}")
        return 0
    else:
        print(f"\nPipeline failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())