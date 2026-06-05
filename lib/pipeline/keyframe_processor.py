#!/usr/bin/env python3
"""
Keyframe Processing for 4D Gaussian Splatting

Combines per-frame point clouds into keyframe NPZ with velocity estimates.
Adapted from FreeTimeGsVanilla's combine_frames_fast_keyframes.py.

Output format:
  - positions: [N, 3] float32 - 3D coordinates
  - velocities: [N, 3] float32 - velocity vectors
  - colors: [N, 3] float32 - RGB colors (0-1)
  - times: [N] float32 - normalized timestamps
  - durations: [N] float32 - temporal duration
  - has_velocity: [N] bool - valid velocity mask
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"[{progress:.1f}%] {message}")


def load_frame_data(input_dir: Path, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load point cloud and colors for a single frame."""
    points_path = input_dir / f"points3d_frame{frame_idx:06d}.npy"
    colors_path = input_dir / f"colors_frame{frame_idx:06d}.npy"

    if not points_path.exists() or not colors_path.exists():
        return np.array([]), np.array([])

    positions = np.load(points_path)
    colors = np.load(colors_path)

    return positions, colors


def compute_velocity_knn(
    pos_t: np.ndarray,
    pos_t1: np.ndarray,
    k: int = 5,
    max_distance: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity estimates using k-NN matching between consecutive frames.

    Args:
        pos_t: [N, 3] positions at time t
        pos_t1: [M, 3] positions at time t+1
        k: number of nearest neighbors
        max_distance: max distance for valid match

    Returns:
        velocities: [N, 3] velocity vectors
        valid_mask: [N] bool mask for valid estimates
    """
    if len(pos_t) == 0 or len(pos_t1) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)

    # Find nearest neighbors in next frame
    nbrs = NearestNeighbors(n_neighbors=min(k, len(pos_t1)), algorithm='ball_tree').fit(pos_t1)
    distances, indices = nbrs.kneighbors(pos_t)

    # Compute velocities for closest matches
    velocities = np.zeros_like(pos_t)
    valid_mask = np.zeros(len(pos_t), dtype=bool)

    for i in range(len(pos_t)):
        if distances[i][0] < max_distance:
            velocities[i] = pos_t1[indices[i][0]] - pos_t[i]
            valid_mask[i] = True

    return velocities, valid_mask


def estimate_scene_scale(positions: List[np.ndarray]) -> float:
    """Estimate scene scale from all positions."""
    all_points = np.vstack(positions)
    center = all_points.mean(axis=0)
    distances = np.linalg.norm(all_points - center, axis=1)
    return np.percentile(distances, 95) + 1e-6


def smart_density_velocity_sampling(
    positions: np.ndarray,
    velocities: np.ndarray,
    colors: np.ndarray,
    times: np.ndarray,
    durations: np.ndarray,
    has_velocity: np.ndarray,
    target_count: int = 100_000,
    voxel_size: float = 0.01,
    velocity_weight: float = 3.0,
    center_weight: float = 2.0
) -> Dict:
    """
    Smart sampling to reduce point count while preserving motion.

    Args:
        positions: [N, 3] 3D positions
        velocities: [N, 3] velocity vectors
        colors: [N, 3] RGB colors
        times: [N] timestamps
        durations: [N] temporal durations
        has_velocity: [N] valid velocity mask
        target_count: target number of points
        voxel_size: voxel grid size
        velocity_weight: weight for velocity points
        center_weight: weight for center points

    Returns:
        Sampled arrays
    """
    n = len(positions)

    # Compute importance scores
    scores = np.ones(n)

    # Higher score for moving points
    if np.any(has_velocity):
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        velocity_score = velocity_magnitude / (velocity_magnitude.max() + 1e-6)
        scores += velocity_weight * velocity_score * has_velocity

    # Higher score for center points
    center = positions.mean(axis=0)
    center_dist = np.linalg.norm(positions - center, axis=1)
    max_dist = center_dist.max() + 1e-6
    center_score = 1 - (center_dist / max_dist)
    scores += center_weight * center_score

    # Voxel-based sampling
    voxel_coords = np.floor(positions / voxel_size).astype(int)
    voxel_keys = set(map(tuple, voxel_coords))

    # Select points from each voxel
    selected = []
    for key in voxel_keys:
        mask = np.all(voxel_coords == np.array(key), axis=1)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            # Pick point with highest score
            best_idx = indices[np.argmax(scores[indices])]
            selected.append(best_idx)

    selected = np.array(selected)

    # If too few, add more randomly
    if len(selected) < target_count:
        remaining = np.setdiff1d(np.arange(n), selected)
        np.random.shuffle(remaining)
        additional = remaining[:min(target_count - len(selected), len(remaining))]
        selected = np.concatenate([selected, additional])

    # If too many, subsample
    if len(selected) > target_count:
        selected = np.random.choice(selected, target_count, replace=False)

    return {
        'positions': positions[selected],
        'velocities': velocities[selected],
        'colors': colors[selected],
        'times': times[selected],
        'durations': durations[selected],
        'has_velocity': has_velocity[selected]
    }


def process_keyframes(
    input_dir: str,
    output_path: str,
    frame_start: int = 0,
    frame_end: int = 60,
    keyframe_step: int = 5,
    target_points: int = 100_000,
    voxel_size: float = 0.05,
    velocity_weight: float = 5.0,
    center_weight: float = 2.0
) -> Dict:
    """
    Process keyframes and create NPZ with velocity estimates.

    Args:
        input_dir: Directory containing point3d_frame*.npy and colors_frame*.npy
        output_path: Output NPZ file path
        frame_start: Starting frame
        frame_end: Ending frame
        keyframe_step: Interval between keyframes
        target_points: Target number of output points
        voxel_size: Voxel size for sampling
        velocity_weight: Weight for velocity in sampling
        center_weight: Weight for center proximity in sampling

    Returns:
        Result dict with success status
    """
    input_dir = Path(input_dir)

    emit("processing_keyframes", 0, "Loading keyframes...")

    # Load keyframes
    keyframes = list(range(frame_start, frame_end, keyframe_step))
    all_positions = []
    all_colors = []
    all_times = []
    all_velocities = []
    all_has_velocity = []

    for i, frame_idx in enumerate(keyframes):
        progress = (i / len(keyframes)) * 30
        emit("processing_keyframes", progress, f"Loading frame {frame_idx}...")

        positions, colors = load_frame_data(input_dir, frame_idx)

        if len(positions) == 0:
            continue

        # Normalize colors to [0, 1]
        if colors.max() > 1:
            colors = colors.astype(np.float32) / 255.0

        all_positions.append(positions)
        all_colors.append(colors)
        all_times.append(np.full(len(positions), frame_idx / frame_end))

        # Compute velocity if we have previous frame
        if i > 0:
            prev_positions = all_positions[i - 1]
            velocities, valid_mask = compute_velocity_knn(
                prev_positions, positions,
                k=3, max_distance=0.5
            )
            all_velocities.append(velocities)
            all_has_velocity.append(valid_mask)
        else:
            all_velocities.append(np.zeros_like(positions))
            all_has_velocity.append(np.zeros(len(positions), dtype=bool))

    if not all_positions:
        return {"success": False, "error": "No keyframes loaded"}

    # Estimate scene scale
    scene_scale = estimate_scene_scale(all_positions)
    emit("processing_keyframes", 35, f"Scene scale: {scene_scale:.3f}m")

    # Concatenate all keyframes
    positions = np.vstack(all_positions)
    colors = np.vstack(all_colors)
    times = np.concatenate(all_times)
    velocities = np.vstack(all_velocities)
    has_velocity = np.concatenate(all_has_velocity)

    # Assign durations based on velocity
    velocity_mag = np.linalg.norm(velocities, axis=1)
    durations = np.ones(len(positions)) * 0.1
    durations[has_velocity] = np.clip(velocity_mag[has_velocity] * 0.5, 0.05, 0.3)

    emit("processing_keyframes", 50, f"Loaded {len(positions)} points from {len(keyframes)} keyframes")

    # Smart sampling
    emit("processing_keyframes", 60, "Smart sampling...")

    sampled = smart_density_velocity_sampling(
        positions, velocities, colors, times, durations, has_velocity,
        target_count=target_points,
        voxel_size=voxel_size,
        velocity_weight=velocity_weight,
        center_weight=center_weight
    )

    emit("processing_keyframes", 80, f"Sampled {len(sampled['positions'])} points")

    # Save NPZ
    np.savez(
        output_path,
        positions=sampled['positions'].astype(np.float32),
        velocities=sampled['velocities'].astype(np.float32),
        colors=sampled['colors'].astype(np.float32),
        times=sampled['times'].astype(np.float32),
        durations=sampled['durations'].astype(np.float32),
        has_velocity=sampled['has_velocity']
    )

    emit("processing_keyframes", 100, f"Saved keyframes to {output_path}")

    return {
        "success": True,
        "output_path": output_path,
        "num_points": len(sampled['positions']),
        "num_keyframes": len(keyframes),
        "keyframes": keyframes,
        "scene_scale": float(scene_scale)
    }


def main():
    parser = argparse.ArgumentParser(description="Keyframe Processing for 4DGS")
    parser.add_argument("--input-dir", required=True, help="Input directory with point cloud frames")
    parser.add_argument("--output-path", required=True, help="Output NPZ file path")
    parser.add_argument("--frame-start", type=int, default=0, help="Starting frame")
    parser.add_argument("--frame-end", type=int, default=60, help="Ending frame")
    parser.add_argument("--keyframe-step", type=int, default=5, help="Keyframe interval")
    parser.add_argument("--target-points", type=int, default=100_000, help="Target number of points")

    args = parser.parse_args()

    result = process_keyframes(
        input_dir=args.input_dir,
        output_path=args.output_path,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        keyframe_step=args.keyframe_step,
        target_points=args.target_points
    )

    if result['success']:
        print(f"\nKeyframe processing complete!")
        print(f"  Points: {result['num_points']}")
        print(f"  Keyframes: {result['num_keyframes']}")
        return 0
    else:
        print(f"\nFailed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())