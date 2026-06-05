#!/usr/bin/env python3
"""
4D Gaussian Splatting Viewer Server

Starts a viser-based 3D/4D viewer for Gaussian splat visualization.
This server provides a web interface for viewing trained models.

Usage:
    python viewer_server.py --ckpt /path/to/checkpoint.pt --port 8080

Requirements:
    - viser
    - torch
    - gsplat (for rendering)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ViewerServer:
    """4D Gaussian Splatting viewer using viser."""

    def __init__(
        self,
        checkpoint_path: str,
        port: int = 8080,
        host: str = "localhost",
        total_frames: int = 60,
        temporal_threshold: float = 0.05,
        spatial_percentile: float = 95
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.port = port
        self.host = host
        self.total_frames = total_frames
        self.temporal_threshold = temporal_threshold
        self.spatial_percentile = spatial_percentile

        self.server = None
        self.checkpoint_data: Optional[Dict] = None

    def load_checkpoint(self) -> bool:
        """Load the Gaussian checkpoint."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {self.checkpoint_path}")
            return False

        try:
            import torch
            self.checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu")
            logger.info(f"Checkpoint loaded: {len(self.checkpoint_data)} entries")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def start_viser(self) -> bool:
        """Start the viser viewer server."""
        try:
            import viser
            import numpy as np
        except ImportError as e:
            logger.error(f"viser not installed: {e}")
            return False

        logger.info("Starting viser server...")

        # Create server
        self.server = viser.ViserServer(host=self.host, port=self.port)

        # Get Gaussian data
        means = self.checkpoint_data.get("means", None)
        if means is None:
            logger.error("No means found in checkpoint")
            return False

        # Create scene
        scene = self.server.scene

        # Create Gaussian point cloud
        num_points = len(means)

        # Get other Gaussian properties
        scales = self.checkpoint_data.get("scales", torch.ones(num_points, 3))
        quats = self.checkpoint_data.get("quats", torch.randn(num_points, 4))
        quats = quats / quats.norm(dim=-1, keepdim=True)
        opacities = self.checkpoint_data.get("opacities", torch.ones(num_points))
        rgbs = self.checkpoint_data.get("rgbs", torch.rand(num_points, 3))

        # Normalize colors to [0, 1]
        if rgbs.max() > 1:
            rgbs = rgbs / 255.0

        # Create GUI controls
        with self.server.gui.add_folder("Playback"):
            time_slider = self.server.gui.add_slider(
                "Time", min_value=0, max_value=self.total_frames - 1, step=1, initial_value=0
            )
            play_button = self.server.gui.add_button("Play/Pause")
            reset_button = self.server.gui.add_button("Reset View")

        with self.server.gui.add_folder("Visualization"):
            thresh_slider = self.server.gui.add_slider(
                "Temporal Threshold", min_value=0, max_value=0.5, step=0.01,
                initial_value=self.temporal_threshold
            )
            opacity_slider = self.server.gui.add_slider(
                "Point Opacity", min_value=0, max_value=1, step=0.1, initial_value=0.8
            )
            point_size_slider = self.server.gui.add_slider(
                "Point Size", min_value=0.001, max_value=0.1, step=0.001, initial_value=0.02
            )

        with self.server.gui.add_folder("Camera"):
            fov_slider = self.server.gui.add_slider(
                "FOV", min_value=10, max_value=120, step=1, initial_value=50
            )

        # Add stats panel
        self.server.gui.add_text("Gaussians").value = f"{num_points:,}"
        self.server.gui.add_text("Frames").value = str(self.total_frames)

        # State
        is_playing = [False]
        current_time = [0]

        def toggle_play(_):
            is_playing[0] = not is_playing[0]

        def reset_view(_):
            # Reset camera to default position
            pass

        play_button.on_click(toggle_play)
        reset_button.on_click(reset_view)

        logger.info(f"Viewer started at http://{self.host}:{self.port}")
        logger.info("Use the GUI to control playback and visualization")

        # Keep server running
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down viewer...")
            return True

    def start_demo(self) -> bool:
        """Start a demo viewer without checkpoint."""
        try:
            import viser
            import numpy as np
        except ImportError as e:
            logger.error(f"viser not installed: {e}")
            return False

        logger.info("Starting demo viewer...")

        # Create server
        self.server = viser.ViserServer(host=self.host, port=self.port)

        # Generate demo Gaussian data
        np.random.seed(42)
        num_points = 50000

        # Create a sphere with colors
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = 2 + np.random.normal(0, 0.1, num_points)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        # Color based on position
        colors = np.zeros((num_points, 3))
        colors[:, 0] = 0.5 + 0.5 * np.sin(theta)
        colors[:, 1] = 0.5 + 0.5 * np.sin(phi)
        colors[:, 2] = 0.5 + 0.5 * np.cos(theta)

        # Create point cloud
        points = np.stack([x, y, z], axis=1).astype(np.float32)

        # Add GUI controls
        with self.server.gui.add_folder("Playback"):
            time_slider = self.server.gui.add_slider(
                "Time", min_value=0, max_value=self.total_frames - 1, step=1, initial_value=0
            )
            play_button = self.server.gui.add_button("Play/Pause")

        with self.server.gui.add_folder("Visualization"):
            thresh_slider = self.server.gui.add_slider(
                "Temporal Threshold", min_value=0, max_value=0.5, step=0.01,
                initial_value=self.temporal_threshold
            )
            opacity_slider = self.server.gui.add_slider(
                "Point Opacity", min_value=0, max_value=1, step=0.1, initial_value=0.8
            )

        # Add info
        self.server.gui.add_text("Points").value = f"{num_points:,}"
        self.server.gui.add_text("Mode").value = "Demo (no checkpoint)"

        logger.info(f"Demo viewer started at http://{self.host}:{self.port}")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down viewer...")
            return True

    def run(self) -> bool:
        """Run the viewer server."""
        logger.info("=" * 50)
        logger.info("Starting 4D Gaussian Splatting Viewer")
        logger.info("=" * 50)

        # Load checkpoint or start demo
        if self.checkpoint_path.exists():
            if self.load_checkpoint():
                return self.start_viser()
            else:
                logger.warning("Failed to load checkpoint, starting demo mode")
                return self.start_demo()
        else:
            logger.info("No checkpoint found, starting demo mode")
            return self.start_demo()


def main():
    parser = argparse.ArgumentParser(
        description="Start 4D Gaussian Splatting viewer server"
    )
    parser.add_argument(
        "--ckpt",
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viewer server"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for viewer server"
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=60,
        help="Total number of temporal frames"
    )
    parser.add_argument(
        "--temporal-threshold",
        type=float,
        default=0.05,
        help="Temporal opacity threshold"
    )
    parser.add_argument(
        "--spatial-percentile",
        type=float,
        default=95,
        help="Percentile of points to keep"
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

    server = ViewerServer(
        checkpoint_path=args.ckpt or "",
        port=args.port,
        host=args.host,
        total_frames=args.total_frames,
        temporal_threshold=args.temporal_threshold,
        spatial_percentile=args.spatial_percentile
    )

    result = server.run()
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()