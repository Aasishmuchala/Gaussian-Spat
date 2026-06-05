#!/usr/bin/env python3
"""
COLMAP SfM Pipeline for 4D Gaussian Splatting

Runs COLMAP Structure-from-Motion on extracted frames to produce
sparse 3D point clouds and camera poses required for 4DGS.

Usage:
    python colmap_processor.py --frames-dir /path/to/frames --output-dir /path/to/output

Requirements:
    - COLMAP installed (https://colmap.github.io/)
    - Optional: pycolmap for Python bindings
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class COLMAPProcessor:
    """Handles COLMAP SfM processing for 4DGS."""

    def __init__(
        self,
        frames_dir: str,
        output_dir: str,
        single_camera: bool = True,
        max_num_features: int = 8192,
        use_gpu: bool = True
    ):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.single_camera = single_camera
        self.max_num_features = max_num_features
        self.use_gpu = use_gpu

        # Create output directories
        self.sparse_dir = self.output_dir / "sparse"
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.per_frame_dir = self.output_dir / "per_frame_points"
        self.per_frame_dir.mkdir(parents=True, exist_ok=True)

        self.database_path = self.output_dir / "database.db"

        # Metadata storage
        self.metadata = {
            "frames_dir": str(self.frames_dir),
            "sparse_dir": str(self.sparse_dir),
            "num_cameras": 0,
            "num_frames": 0,
            "num_points3d": 0
        }

    def check_colmap(self) -> bool:
        """Check if COLMAP is installed."""
        try:
            result = subprocess.run(
                ['colmap', '--version'],
                capture_output=True,
                text=True
            )
            logger.info(f"COLMAP version: {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            logger.error("COLMAP not found. Please install COLMAP first.")
            logger.info("Download from: https://demuc.de/colmap/")
            return False

    def get_camera_directories(self) -> List[Path]:
        """Get list of camera directories."""
        return sorted([d for d in self.frames_dir.iterdir() if d.is_dir()])

    def count_frames(self) -> int:
        """Count total frames across all cameras."""
        total = 0
        for cam_dir in self.get_camera_directories():
            frames = list(cam_dir.glob("frame_*.jpg"))
            total += len(frames)
        return total

    def create_image_list(self) -> Path:
        """
        Create a single merged image list from all camera directories.
        This is needed for COLMAP to process multi-view data.
        """
        image_list_path = self.output_dir / "image_list.txt"

        with open(image_list_path, 'w') as f:
            for cam_dir in self.get_camera_directories():
                for img in sorted(cam_dir.glob("frame_*.jpg")):
                    f.write(str(img) + '\n')

        logger.info(f"Created image list with {self.count_frames()} images")
        return image_list_path

    def run_feature_extraction(self) -> bool:
        """Run SIFT feature extraction."""
        logger.info("Running COLMAP feature extraction...")

        # Use a temp directory approach - create a merged images folder
        merged_images = self.output_dir / "images_merged"
        merged_images.mkdir(exist_ok=True)

        # Copy all images with unique naming
        cam_idx = 0
        for cam_dir in self.get_camera_directories():
            for img in sorted(cam_dir.glob("frame_*.jpg")):
                new_name = f"cam{cam_idx:02d}_{img.name}"
                shutil.copy2(img, merged_images / new_name)
            cam_idx += 1

        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(merged_images),
            '--ImageReader.single_camera', '1' if self.single_camera else '0',
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
            '--FeatureExtraction.max_num_features', str(self.max_num_features),
            '--FeatureExtraction.use_gpu', '1' if self.use_gpu else '0',
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Feature extraction failed:\n{result.stderr}")
                return False

            logger.info("Feature extraction complete")
            return True

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return False

    def run_matching(self) -> bool:
        """Run feature matching."""
        logger.info("Running COLMAP feature matching...")

        cmd = [
            'colmap', 'exhaustive_matcher',
            '--database_path', str(self.database_path),
            '--SiftMatching.use_gpu', '1' if self.use_gpu else '0',
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.warning(f"Matching stderr:\n{result.stderr}")

            logger.info("Feature matching complete")
            return True

        except Exception as e:
            logger.error(f"Matching error: {e}")
            return False

    def run_sfm(self) -> bool:
        """Run incremental Structure-from-Motion."""
        logger.info("Running COLMAP incremental SfM...")

        # Ensure sparse directory exists
        model_dir = self.sparse_dir / "0"
        model_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'colmap', 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.output_dir / "images_merged"),
            '--output_path', str(self.sparse_dir),
            '--Mapper.max_num_models', '1',
            '--Mapper.max_model_size', '50000',
            '--Mapper.init_min_tri_angle', '16.0',
            '--Mapper.num_workers', '4',
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"SfM failed:\n{result.stderr}")
                return False

            logger.info("Incremental SfM complete")
            return True

        except Exception as e:
            logger.error(f"SfM error: {e}")
            return False

    def read_sparse_model(self) -> Optional[Dict]:
        """
        Read the sparse reconstruction model.
        Uses pycolmap if available, otherwise uses text parsing.
        """
        try:
            import pycolmap

            model_path = self.sparse_dir / "0"
            if not model_path.exists():
                model_path = self.sparse_dir  # Try parent

            if model_path.exists():
                reconstruction = pycolmap.Reconstruction(str(model_path))
                return {
                    "num_cameras": len(reconstruction.cameras),
                    "num_images": len(reconstruction.images),
                    "num_points3d": len(reconstruction.points3D)
                }
        except ImportError:
            logger.info("pycolmap not available, using text parsing")

        # Text parsing fallback
        cameras_txt = self.sparse_dir / "0" / "cameras.txt"
        if cameras_txt.exists():
            with open(cameras_txt) as f:
                lines = [l for l in f.readlines() if not l.startswith('#')]
                self.metadata['num_cameras'] = len(lines)

        images_txt = self.sparse_dir / "0" / "images.txt"
        if images_txt.exists():
            with open(images_txt) as f:
                lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
                # Each image is 2 lines
                self.metadata['num_frames'] = len(lines) // 2

        points3d_txt = self.sparse_dir / "0" / "points3D.txt"
        if points3d_txt.exists():
            with open(points3d_txt) as f:
                lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
                self.metadata['num_points3d'] = len(lines)

        return self.metadata

    def extract_per_frame_points(self) -> bool:
        """
        Extract per-frame visible 3D points as NumPy arrays.
        This creates the point clouds needed for 4DGS training.
        """
        logger.info("Extracting per-frame point clouds...")

        try:
            import pycolmap
            import numpy as np
        except ImportError as e:
            logger.warning(f"Cannot extract per-frame points: {e}")
            logger.info("Install pycolmap and numpy for this feature")
            return False

        try:
            model_path = self.sparse_dir / "0"
            reconstruction = pycolmap.Reconstruction(str(model_path))

            # Process each image
            for image_id, image in reconstruction.images.items():
                visible_points = []
                visible_colors = []

                for point2D in image.points2D:
                    if point2D.has_point3D:
                        pt3D = reconstruction.points3D[point2D.point3D_id]
                        visible_points.append(pt3D.xyz)
                        visible_colors.append(pt3D.rgb / 255.0)  # Normalize to [0, 1]

                if visible_points:
                    points_array = np.array(visible_points, dtype=np.float32)
                    colors_array = np.array(visible_colors, dtype=np.float32)

                    # Save as NPZ
                    frame_path = self.per_frame_dir / f"frame_{image_id:04d}.npz"
                    np.savez_compressed(
                        frame_path,
                        positions=points_array,
                        colors=colors_array
                    )

            logger.info(f"Extracted {len(reconstruction.images)} frame point clouds")
            return True

        except Exception as e:
            logger.error(f"Per-frame extraction error: {e}")
            return False

    def run(self) -> Dict:
        """Run the complete COLMAP pipeline."""
        logger.info("=" * 50)
        logger.info("Starting COLMAP SfM Pipeline")
        logger.info("=" * 50)

        # Check COLMAP installation
        if not self.check_colmap():
            return {"success": False, "error": "COLMAP not installed"}

        # Count frames
        num_frames = self.count_frames()
        logger.info(f"Processing {num_frames} frames from {len(self.get_camera_directories())} cameras")

        # Step 1: Feature extraction
        if not self.run_feature_extraction():
            return {"success": False, "error": "Feature extraction failed"}

        # Step 2: Feature matching
        if not self.run_matching():
            return {"success": False, "error": "Feature matching failed"}

        # Step 3: Incremental SfM
        if not self.run_sfm():
            return {"success": False, "error": "Incremental SfM failed"}

        # Step 4: Read sparse model
        model_info = self.read_sparse_model()

        # Step 5: Extract per-frame points
        self.extract_per_frame_points()

        # Save metadata
        self.metadata.update(model_info)
        metadata_path = self.output_dir / "colmap_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("=" * 50)
        logger.info(f"COLMAP processing complete!")
        logger.info(f"  Cameras: {self.metadata.get('num_cameras', 'N/A')}")
        logger.info(f"  Frames: {self.metadata.get('num_frames', 'N/A')}")
        logger.info(f"  3D Points: {self.metadata.get('num_points3d', 'N/A')}")
        logger.info(f"  Sparse model: {self.sparse_dir / '0'}")
        logger.info("=" * 50)

        return {
            "success": True,
            "sparse_dir": str(self.sparse_dir),
            "per_frame_dir": str(self.per_frame_dir),
            "metadata": self.metadata
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM on extracted video frames"
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory containing extracted frames"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for COLMAP results"
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        default=True,
        help="Assume single shared camera model"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=8192,
        help="Maximum number of SIFT features per image"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    processor = COLMAPProcessor(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        single_camera=args.single_camera,
        max_num_features=args.max_features,
        use_gpu=not args.cpu
    )

    result = processor.run()
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()