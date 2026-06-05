#!/usr/bin/env python3
"""
Video Frame Extraction Pipeline

Extracts synchronized frames from multiple video files using FFmpeg.
This script handles multi-view video input and outputs aligned frame sequences
for COLMAP processing.

Usage:
    python video_processor.py --input-dir /path/to/videos --output-dir /path/to/output --fps 2

Requirements:
    - FFmpeg installed and in PATH
    - OpenCV for video reading (optional, FFmpeg is preferred)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame extraction and synchronization."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        fps: float = 2.0,
        max_frames: Optional[int] = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.max_frames = max_frames

        # Create output directories
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata = {
            "fps": fps,
            "cameras": {},
            "total_frames": 0
        }

    def find_video_files(self) -> List[Path]:
        """Find all video files in the input directory."""
        video_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.mpeg', '.mpg'}
        videos = []

        for ext in video_extensions:
            videos.extend(self.input_dir.glob(f"*{ext}"))
            videos.extend(self.input_dir.glob(f"*{ext.upper()}"))

        # Also check for camera_* pattern
        for pattern in ['camera_*.mp4', 'cam_*.mp4', 'video_*.mp4']:
            videos.extend(self.input_dir.glob(pattern))

        return sorted(set(videos))

    def get_video_info(self, video_path: Path) -> Dict:
        """Get video metadata using FFprobe."""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFprobe failed for {video_path}: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FFprobe output: {e}")
            return {}

    def extract_frames(
        self,
        video_path: Path,
        camera_name: str,
        frame_offset: int = 0
    ) -> Tuple[int, int]:
        """
        Extract frames from a single video using FFmpeg.

        Returns:
            Tuple of (num_frames_extracted, frame_width, frame_height)
        """
        # Create camera-specific output directory
        camera_dir = self.frames_dir / camera_name
        camera_dir.mkdir(exist_ok=True)

        # FFmpeg command for frame extraction
        # Using select filter to extract frames at specified FPS
        output_pattern = str(camera_dir / "frame_%06d.jpg")

        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'fps={self.fps}',
            '-q:v', '2',  # JPEG quality
            '-vsync', '0',  # Don't drop frames
        ]

        if self.max_frames:
            cmd.extend(['-frames:v', str(self.max_frames)])

        cmd.append(output_pattern)

        logger.info(f"Extracting frames from {video_path.name}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.warning(f"FFmpeg stderr: {result.stderr}")

            # Count extracted frames
            extracted_files = list(camera_dir.glob("frame_*.jpg"))
            num_frames = len(extracted_files)

            # Get video dimensions
            info = self.get_video_info(video_path)
            width = height = 0
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    break

            logger.info(f"Extracted {num_frames} frames from {video_path.name}")

            return num_frames, width, height

        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            return 0, 0, 0

    def sync_frames(self, camera_dirs: List[Path]) -> None:
        """
        Synchronize frames across cameras by finding common timestamps.
        For synchronized capture, frames should already be aligned.
        """
        logger.info("Synchronizing frames across cameras...")

        # If cameras are truly synchronized, frames are already aligned
        # This method can be used for manual sync if needed

        # Find minimum frame count
        min_frames = float('inf')
        for cam_dir in camera_dirs:
            frames = list(cam_dir.glob("frame_*.jpg"))
            min_frames = min(min_frames, len(frames))

        # Optionally remove extra frames for perfect sync
        logger.info(f"Minimum frame count across cameras: {min_frames}")

    def run(self) -> Dict:
        """Run the complete extraction pipeline."""
        logger.info("=" * 50)
        logger.info("Starting Video Frame Extraction Pipeline")
        logger.info("=" * 50)

        # Find video files
        videos = self.find_video_files()

        if not videos:
            logger.error(f"No video files found in {self.input_dir}")
            return {"success": False, "error": "No video files found"}

        logger.info(f"Found {len(videos)} video files")

        # Process each video
        camera_dirs = []
        total_frames = 0

        for idx, video_path in enumerate(videos):
            camera_name = f"camera_{idx:02d}"

            # Get video info
            info = self.get_video_info(video_path)
            duration = 0
            width = height = 0

            if 'format' in info:
                duration = float(info['format'].get('duration', 0))

            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    break

            # Extract frames
            num_frames, vid_width, vid_height = self.extract_frames(video_path, camera_name)

            if num_frames > 0:
                camera_dirs.append(self.frames_dir / camera_name)
                total_frames += num_frames

                self.metadata['cameras'][camera_name] = {
                    "source_video": str(video_path.name),
                    "num_frames": num_frames,
                    "width": vid_width or width,
                    "height": vid_height or height,
                    "duration": duration,
                    "fps": self.fps
                }

        # Synchronize frames
        if camera_dirs:
            self.sync_frames(camera_dirs)

        # Update metadata
        self.metadata['total_frames'] = total_frames
        self.metadata['num_cameras'] = len(camera_dirs)

        # Save metadata
        metadata_path = self.output_dir / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("=" * 50)
        logger.info(f"Extraction complete!")
        logger.info(f"  Total cameras: {len(camera_dirs)}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Output directory: {self.frames_dir}")
        logger.info("=" * 50)

        return {
            "success": True,
            "num_cameras": len(camera_dirs),
            "total_frames": total_frames,
            "output_dir": str(self.frames_dir),
            "metadata_path": str(metadata_path)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract synchronized frames from multi-view video files"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input video files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to extract (default: 2.0)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract per video"
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

    processor = VideoProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames
    )

    result = processor.run()
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()