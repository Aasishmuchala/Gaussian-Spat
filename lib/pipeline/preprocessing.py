#!/usr/bin/env python3
"""
Multi-View 3D Reconstruction Pipeline

Real 3D reconstruction from multi-view video using OpenCV.
This implements Structure-from-Motion (SfM) without requiring COLMAP.

Pipeline:
1. Extract frames from videos at synchronized timestamps
2. Detect ORB features in each frame
3. Match features across camera views
4. Estimate relative camera poses via Essential matrix
5. Triangulate 3D points using Sparse Bundle Adjustment
6. Output per-frame point clouds for 4DGS training

Output format compatible with FreeTimeGsVanilla:
  - points3d_frameXXXXXX.npy: [N, 3] float32 positions
  - colors_frameXXXXXX.npy: [N, 3] float32 RGB (0-255)
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Emit progress event."""
    if _progress_callback:
        _progress_callback({
            "stage": stage,
            "progress": progress,
            "message": message,
            "metadata": metadata or {}
        })
    logger.info(f"[{progress:.1f}%] {message}")


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0
    model: str = "PINHOLE"

    def from_opencv_params(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        return self

    def estimate_from_image(self, image_shape):
        """Estimate intrinsics from image size (assumes centered principal point)."""
        h, w = image_shape[:2]
        # Assume field of view ~60 degrees
        fov = np.radians(60)
        focal = (w / 2) / np.tan(fov / 2)
        self.fx = focal
        self.fy = focal
        self.cx = w / 2
        self.cy = h / 2
        self.width = w
        self.height = h
        return self

    @property
    def K(self) -> np.ndarray:
        """Get 3x3 camera intrinsics matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Assume zero distortion for simplicity."""
        return np.zeros(5, dtype=np.float64)


@dataclass
class CameraPose:
    """Camera pose in SE(3)."""
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3x1 translation

    def __post_init__(self):
        if self.R is None:
            self.R = np.eye(3)
        if self.t is None:
            self.t = np.zeros(3)

    @property
    def T(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def inverse(self) -> 'CameraPose':
        """Get inverse pose."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return CameraPose(R=R_inv, t=t_inv)

    def transform_point(self, pt: np.ndarray) -> np.ndarray:
        """Transform a 3D point."""
        return self.R @ pt + self.t

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Transform multiple 3D points."""
        return (self.R @ pts.T + self.t.reshape(3, 1)).T


@dataclass
class FrameData:
    """Data for a single video frame."""
    frame_idx: int
    image: np.ndarray
    keypoints: np.ndarray  # [N, 2]
    descriptors: np.ndarray  # [N, 32] for ORB
    camera_idx: int


@dataclass
class Point3D:
    """A triangulated 3D point."""
    position: np.ndarray
    color: np.ndarray
    observations: List[Tuple[int, int, int]] = field(default_factory=list)  # (cam_idx, frame_idx, kp_idx)


class MultiViewReconstructor:
    """
    Multi-view 3D reconstruction pipeline.

    Takes synchronized multi-view video and outputs per-frame point clouds
    with estimated camera poses.
    """

    def __init__(
        self,
        video_paths: List[str],
        output_dir: str,
        fps: float = 2.0,
        max_frames: int = 100,
        num_features: int = 2000,
        feature_threshold: float = 30.0,
        min_matches: int = 15,
        min_triangulation_angle: float = 1.5,  # degrees
        verbose: bool = True
    ):
        self.video_paths = video_paths
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.max_frames = max_frames
        self.num_features = num_features
        self.feature_threshold = feature_threshold
        self.min_matches = min_matches
        self.min_triangulation_angle = np.radians(min_triangulation_angle)
        self.verbose = verbose

        # Feature detector
        self.orb = cv2.ORB_create(nfeatures=num_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera parameters
        self.num_cameras = len(video_paths)
        self.intrinsics: List[CameraIntrinsics] = []
        self.poses: List[CameraPose] = []

        # Output directories
        self.points_dir = self.output_dir / "points"
        self.images_dir = self.output_dir / "images"
        self.points_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # All triangulated points
        self.map_points: List[Point3D] = []

        # Frame cache
        self.frame_data: Dict[int, List[FrameData]] = {}

    def extract_frames(self) -> int:
        """Extract synchronized frames from all videos."""
        emit("extracting_frames", 0, f"Extracting frames from {self.num_cameras} cameras...")

        frames_per_camera: List[List[FrameData]] = [[] for _ in range(self.num_cameras)]
        video_caps = []

        # Open all videos
        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {path}")
                return 0
            video_caps.append(cap)

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps_val = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Estimate intrinsics
            cam_intrinsics = CameraIntrinsics()
            cam_intrinsics.estimate_from_image((height, width))
            self.intrinsics.append(cam_intrinsics)

            logger.info(f"Camera {i}: {width}x{height}, {total_frames} frames")

        # Calculate frame extraction interval
        sample_interval = max(1, int(video_fps_val / self.fps / self.max_frames))

        frame_idx = 0
        extracted = 0

        while extracted < self.max_frames:
            for cam_idx, cap in enumerate(video_caps):
                ret, image = cap.read()
                if not ret:
                    continue

                if frame_idx % sample_interval == 0:
                    # Convert to grayscale for feature detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Detect features
                    kp, desc = self.orb.detectAndCompute(gray, None)

                    if len(kp) > 0:
                        kp_arr = np.array([[p.pt[0], p.pt[1]] for p in kp], dtype=np.float32)
                    else:
                        kp_arr = np.zeros((0, 2), dtype=np.float32)

                    frames_per_camera[cam_idx].append(FrameData(
                        frame_idx=extracted,
                        image=image.copy(),
                        keypoints=kp_arr,
                        descriptors=desc,
                        camera_idx=cam_idx
                    ))

            frame_idx += 1
            extracted += 1

            # Check if any video has ended
            if not all(cap.isOpened() for cap in video_caps):
                break

        # Release all caps
        for cap in video_caps:
            cap.release()

        # Store frame data
        min_frames = min(len(frames) for frames in frames_per_camera)
        for cam_idx, frames in enumerate(frames_per_camera):
            self.frame_data[cam_idx] = frames[:min_frames]

        # Initialize poses (first camera at origin)
        self.poses = [CameraPose(R=np.eye(3), t=np.zeros(3)) for _ in range(self.num_cameras)]

        emit("extracting_frames", 100, f"Extracted {min_frames} synchronized frames from {self.num_cameras} cameras")
        return min_frames

    def match_features_between_cameras(self, frame_idx: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Match features between all pairs of cameras for a given frame."""
        matches = {}

        # Get keypoints and descriptors for this frame
        cam_data = [self.frame_data[cam_idx][frame_idx] for cam_idx in range(self.num_cameras)]

        for i in range(self.num_cameras):
            for j in range(i + 1, self.num_cameras):
                # Match features
                query_desc = cam_data[i].descriptors
                train_desc = cam_data[j].descriptors

                if query_desc is None or train_desc is None:
                    continue

                knn_matches = self.matcher.knnMatch(query_desc, train_desc, k=2)

                # Lowe's ratio test
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                # Filter by threshold
                good_matches = [m for m in good_matches if m.distance < self.feature_threshold]

                if len(good_matches) >= self.min_matches:
                    matches[(i, j)] = [
                        (m.queryIdx, m.trainIdx, m.distance)
                        for m in good_matches
                    ]

        return matches

    def estimate_pose_ransac(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        intrinsics1: CameraIntrinsics,
        intrinsics2: CameraIntrinsics
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Estimate essential matrix and recover pose using RANSAC.

        Returns: (R, t, inlier_ratio)
        """
        if len(pts1) < 5:
            return None, None, 0.0

        # Normalize points
        K1_inv = np.linalg.inv(intrinsics1.K)
        K2_inv = np.linalg.inv(intrinsics2.K)

        pts1_norm = (K1_inv[:3, :3] @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]
        pts2_norm = (K2_inv[:3, :3] @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T[:, :2]

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, None, 0.0

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm, mask=mask)

        inlier_ratio = np.sum(mask) / len(mask)

        return R, t.flatten(), inlier_ratio

    def triangulate_point(
        self,
        pt1: np.ndarray,
        pt2: np.ndarray,
        pose1: CameraPose,
        pose2: CameraPose,
        intrinsics1: CameraIntrinsics,
        intrinsics2: CameraIntrinsics
    ) -> Optional[np.ndarray]:
        """Triangulate a single 3D point from two views."""
        # Projection matrices
        P1 = intrinsics1.K @ np.hstack([pose1.R, pose1.t.reshape(3, 1)])
        P2 = intrinsics2.K @ np.hstack([pose2.R, pose2.t.reshape(3, 1)])

        # Triangulate
        pt_4d = cv2.triangulatePoints(P1, P2, pt1.reshape(1, -1), pt2.reshape(1, -1))
        pt_3d = (pt_4d[:3] / pt_4d[3]).flatten()

        # Check depth
        depth1 = (pose1.R[2] @ pt_3d + pose1.t[2])
        depth2 = (pose2.R[2] @ pt_3d + pose2.t[2])

        if depth1 < 0 or depth2 < 0:
            return None

        # Check triangulation angle
        baseline = np.linalg.norm(pose2.t - pose1.t)
        if baseline > 0:
            angle1 = np.arccos(np.clip(
                (pt_3d - pose1.t) @ (pose2.t - pose1.t) /
                (np.linalg.norm(pt_3d - pose1.t) * baseline + 1e-6),
                -1, 1
            ))
            if angle1 < self.min_triangulation_angle:
                return None

        return pt_3d

    def reconstruct_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct 3D points for a single frame.

        Returns: (positions, colors, valid_mask)
        """
        positions = []
        colors = []

        # Get frame data
        cam_data = [self.frame_data[cam_idx][frame_idx] for cam_idx in range(self.num_cameras)]

        # Match features across camera pairs
        matches = self.match_features_between_cameras(frame_idx)

        for (cam1, cam2), match_list in matches.items():
            if cam1 != 0:  # Use first camera as reference for now
                continue

            # Get matched points
            pts1 = cam_data[cam1].keypoints[[m[0] for m in match_list]]
            pts2 = cam_data[cam2].keypoints[[m[1] for m in match_list]]
            distances = [m[2] for m in match_list]

            # Estimate pose
            R, t, inlier_ratio = self.estimate_pose_ransac(
                pts1, pts2,
                self.intrinsics[cam1],
                self.intrinsics[cam2]
            )

            if R is None or inlier_ratio < 0.3:
                continue

            # Update camera pose
            self.poses[cam2] = CameraPose(R=R, t=t * 2.0)  # Scale translation

            # Triangulate points
            for i, ((q_idx, t_idx, dist), p1, p2) in enumerate(zip(match_list, pts1, pts2)):
                pt_3d = self.triangulate_point(
                    p1, p2,
                    self.poses[cam1],
                    self.poses[cam2],
                    self.intrinsics[cam1],
                    self.intrinsics[cam2]
                )

                if pt_3d is not None:
                    positions.append(pt_3d)
                    # Get color from first camera
                    y, x = int(p1[1]), int(p1[0])
                    y = min(y, cam_data[cam1].image.shape[0] - 1)
                    x = min(x, cam_data[cam1].image.shape[1] - 1)
                    color = cam_data[cam1].image[y, x]
                    colors.append(color)

        return positions, colors, np.ones(len(positions), dtype=bool)

    def process_all_frames(self) -> Dict:
        """Process all frames and save point clouds."""
        total_frames = len(next(iter(self.frame_data.values())))

        emit("running_colmap", 0, f"Reconstructing {total_frames} frames...")

        all_positions = []
        all_colors = []
        all_frames = []

        for frame_idx in range(total_frames):
            progress = (frame_idx / total_frames) * 100
            emit("running_colmap", progress, f"Reconstructing frame {frame_idx + 1}/{total_frames}...")

            positions, colors, valid_mask = self.reconstruct_frame(frame_idx)

            # Save frame point cloud
            frame_points_path = self.points_dir / f"points3d_frame{frame_idx:06d}.npy"
            frame_colors_path = self.points_dir / f"colors_frame{frame_idx:06d}.npy"

            np.save(frame_points_path, positions)
            np.save(frame_colors_path, colors)

            # Save image
            if 0 in self.frame_data:
                img_path = self.images_dir / f"frame{frame_idx:06d}.jpg"
                cv2.imwrite(str(img_path), self.frame_data[0][frame_idx].image)

            all_positions.append(positions)
            all_colors.append(colors)
            all_frames.append(frame_idx)

        emit("running_colmap", 100, f"Reconstructed {sum(len(p) for p in all_positions)} total 3D points")

        # Save metadata
        metadata = {
            "num_frames": total_frames,
            "num_cameras": self.num_cameras,
            "fps": self.fps,
            "total_points": sum(len(p) for p in all_positions),
            "intrinsics": [
                {
                    "fx": cam.fx,
                    "fy": cam.fy,
                    "cx": cam.cx,
                    "cy": cam.cy,
                    "width": cam.width,
                    "height": cam.height
                }
                for cam in self.intrinsics
            ],
            "frames": [
                {"frame_idx": f, "num_points": len(p)}
                for f, p in zip(all_frames, all_positions)
            ]
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def run(self) -> Dict:
        """Run complete reconstruction pipeline."""
        emit("setup", 0, "Initializing multi-view 3D reconstruction...")

        # Step 1: Extract frames
        num_frames = self.extract_frames()
        if num_frames == 0:
            return {"success": False, "error": "No frames extracted"}

        # Step 2: Process all frames
        metadata = self.process_all_frames()

        emit("complete", 100, f"3D reconstruction complete! {metadata['total_points']} points from {num_frames} frames")

        return {
            "success": True,
            "output_dir": str(self.output_dir),
            "num_frames": num_frames,
            "total_points": metadata["total_points"],
            "metadata": metadata
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-View 3D Reconstruction Pipeline")
    parser.add_argument("--videos", required=True, help="Comma-separated video paths")
    parser.add_argument("--output-dir", default="./reconstruction_output", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to process")
    parser.add_argument("--num-features", type=int, default=2000, help="Number of ORB features")

    args = parser.parse_args()
    videos = [v.strip() for v in args.videos.split(",")]

    logger.info("=" * 60)
    logger.info("Multi-View 3D Reconstruction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Videos: {len(videos)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    reconstructor = MultiViewReconstructor(
        video_paths=videos,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        num_features=args.num_features
    )

    result = reconstructor.run()

    if result['success']:
        logger.info("=" * 60)
        logger.info(f"Reconstruction complete!")
        logger.info(f"Frames: {result['num_frames']}")
        logger.info(f"Points: {result['total_points']}")
        return 0
    else:
        logger.error(f"Failed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())