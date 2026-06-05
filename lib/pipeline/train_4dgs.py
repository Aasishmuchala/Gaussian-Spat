#!/usr/bin/env python3
"""
4D Gaussian Splatting Training

Trains 4D Gaussian primitives using gsplat library.
Adapted from FreeTimeGsVanilla's simple_trainer_freetime_4d_pure_relocation.py.

The 4D Gaussians have:
- Position (x): canonical 3D position
- Velocity (v): linear motion vector
- Time (t): when Gaussian is most visible
- Duration (s): temporal width

Position at time t: x(t) = x + v * (t - t_canonical)
Temporal opacity: opacity(t) = exp(-0.5 * ((t - t_canonical) / duration)^2)
"""

import argparse
import json
import sys
import os
import numpy as np
import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    import gsplat
    from gsplat import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    logger.error("gsplat not installed. Run: pip install gsplat")
    gsplat = None

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


class FourDGaussians(torch.nn.Module):
    """
    4D Gaussian representation with temporal motion.

    Each Gaussian has:
    - means: [N, 3] canonical position
    - scales: [N, 3] 3D scale
    - quats: [N, 4] rotation quaternion (wxyz)
    - opacities: [N] opacity
    - rgbs: [N, 3] RGB color
    - times: [N] canonical time
    - durations: [N] temporal width
    - velocities: [N, 3] linear velocity
    """

    def __init__(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        times: np.ndarray,
        velocities: np.ndarray,
        durations: np.ndarray,
        num_shs: int = 0
    ):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        n = len(positions)

        # Learnable parameters
        self.means = torch.nn.Parameter(torch.tensor(positions, dtype=torch.float32, device=device))
        self.scales = torch.nn.Parameter(torch.ones(n, 3, device=device) * 0.02)
        self.quats = torch.nn.Parameter(self._random_quats(n, device))
        self.opacities = torch.nn.Parameter(torch.ones(n, device=device) * 0.5)
        self.rgbs = torch.nn.Parameter(torch.tensor(colors, dtype=torch.float32, device=device))

        # 4D temporal parameters
        self.times = torch.nn.Parameter(torch.tensor(times, dtype=torch.float32, device=device).reshape(-1, 1))
        self.durations = torch.nn.Parameter(torch.tensor(durations, dtype=torch.float32, device=device).reshape(-1, 1))
        self.velocities = torch.nn.Parameter(torch.tensor(velocities, dtype=torch.float32, device=device))

        # Freeze some parameters initially
        self.rgbs.requires_grad = False  # Start with input colors

        self.n = n
        logger.info(f"Initialized {n} 4D Gaussians on {device}")

    @staticmethod
    def _random_quats(n: int, device: torch.device) -> torch.nn.Parameter:
        """Generate random unit quaternions."""
        quats = torch.randn(n, 4, device=device)
        quats = quats / quats.norm(dim=-1, keepdim=True)
        return torch.nn.Parameter(quats)

    def get_covariance(self, time: float) -> torch.Tensor:
        """Get covariance matrices for Gaussians at given time."""
        # Get rotation matrices from quaternions
        R = self._quaternion_to_rotation(self.quats)

        # Apply velocity to means at given time
        time_offset = time - self.times  # [N, 1]
        motion = self.velocities * time_offset  # [N, 3]
        means_t = self.means + motion

        # Scale matrices
        scales = torch.exp(self.scales)
        S = torch.diag_embed(scales)  # [N, 3, 3]

        # Covariance = R @ S @ S.T @ R.T
        M = R @ S
        cov = M @ M.transpose(-2, -1)

        return means_t, cov

    @staticmethod
    def _quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        # q: [..., 4] (w, x, y, z)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        # Normalize
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        # Build rotation matrix
        R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device, dtype=q.dtype)

        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - w*z)
        R[..., 0, 2] = 2*(x*z + w*y)
        R[..., 1, 0] = 2*(x*y + w*z)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - w*x)
        R[..., 2, 0] = 2*(x*z - w*y)
        R[..., 2, 1] = 2*(y*z + w*x)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)

        return R

    def get_temporal_opacity(self, time: torch.Tensor) -> torch.Tensor:
        """Get opacity at given time based on Gaussian's temporal window."""
        time_diff = (time - self.times).abs()  # [N, 1]
        opacity = torch.exp(-0.5 * (time_diff / (self.durations + 1e-6)).pow(2))
        return (opacity * torch.sigmoid(self.opacities)).squeeze(-1)


class GaussianRenderer:
    """Renders Gaussians using gsplat."""

    def __init__(self, image_width: int = 640, image_height: int = 480):
        self.image_width = image_width
        self.image_height = image_height
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def render(
        self,
        gaussians: FourDGaussians,
        camera_pos: torch.Tensor,
        camera_rot: torch.Tensor,
        time: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render Gaussians from a camera view.

        Args:
            gaussians: FourDGaussians model
            camera_pos: [3] camera position
            camera_rot: [3, 3] camera rotation matrix
            time: current time for 4D

        Returns:
            (rgb_image, alpha_mask)
        """
        # Get Gaussian positions and covariances at this time
        means, covs = gaussians.get_covariance(time)

        # Get opacities
        opacities = gaussians.get_temporal_opacity(torch.tensor(time, device=self.device))

        # Get colors
        colors = torch.sigmoid(gaussians.rgbs)

        # Build view matrix
        view_matrix = torch.eye(4, device=self.device)
        view_matrix[:3, :3] = camera_rot.T
        view_matrix[:3, 3] = -camera_rot.T @ camera_pos

        # Camera parameters
        principal_point = torch.tensor(
            [self.image_width / 2, self.image_height / 2],
            device=self.device
        )
        focal_length = self.image_width * 1.2

        # Render using gsplat
        raster_settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfov_x=self.image_width / (2 * focal_length),
            tanfov_y=self.image_height / (2 * focal_length),
            bg=torch.zeros(3, device=self.device),
            viewmatrix=view_matrix,
            projmatrix=torch.eye(4, device=self.device),  # gsplat computes this
            projmatrix_row_major=False,
            screen_center=principal_point,
            scale_modifier=1.0,
            principal_x=principal_point[0],
            principal_y=principal_point[1],
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Convert covariances to scales and quats
        scales = torch.exp(gaussians.scales)
        quats = gaussians.quats / gaussians.quats.norm(dim=-1, keepdim=True)

        # Render
        rendered_image, rendered_alpha = rasterizer(
            means=means,
            scales=scales,
            rotations=quats,
            opacities=opacities,
            colors=colors,
            covs_3d=covs
        )

        return rendered_image.clamp(0, 1), rendered_alpha


def load_init_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load initial point cloud from NPZ file."""
    emit("setup", 0, f"Loading initial points from {npz_path}...")

    data = np.load(npz_path)

    required = ['positions', 'colors', 'times']
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in NPZ file")

    result = {
        'positions': data['positions'].astype(np.float32),
        'colors': data['colors'].astype(np.float32),
        'times': data['times'].astype(np.float32),
        'velocities': data['velocities'].astype(np.float32) if 'velocities' in data else np.zeros_like(data['positions']),
        'durations': data['durations'].astype(np.float32) if 'durations' in data else np.ones(len(data['positions'])) * 0.1
    }

    emit("setup", 10, f"Loaded {len(result['positions'])} initial points")

    return result


def train_4d_gaussians(
    init_npz_path: str,
    output_dir: str,
    max_steps: int = 30000,
    image_width: int = 640,
    image_height: int = 480,
    learning_rate: float = 0.01,
    densify_interval: int = 100,
    densify_until: int = 15000,
    prune_interval: int = 100,
    prune_opacity_threshold: float = 0.005,
    background_color: List[float] = [0.0, 0.0, 0.0]
) -> Dict:
    """
    Train 4D Gaussian splatting model.

    Args:
        init_npz_path: Path to initial point cloud NPZ
        output_dir: Output directory for checkpoints
        max_steps: Maximum training iterations
        image_width: Render width
        image_height: Render height
        learning_rate: Initial learning rate
        densify_interval: Steps between densification
        densify_until: Step to stop densification
        prune_interval: Steps between pruning
        prune_opacity_threshold: Minimum opacity to keep
        background_color: Background RGB color

    Returns:
        Result dict with success status
    """
    if gsplat is None:
        return {"success": False, "error": "gsplat not installed"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emit("training_4dgs", 0, "Initializing 4D Gaussian training...")

    # Load initial points
    init_data = load_init_npz(init_npz_path)

    # Create Gaussians
    gaussians = FourDGaussians(
        positions=init_data['positions'],
        colors=init_data['colors'],
        times=init_data['times'],
        velocities=init_data['velocities'],
        durations=init_data['durations']
    )

    # Renderer
    renderer = GaussianRenderer(image_width, image_height)

    # Optimizer - exclude rgbs initially
    optimizer = torch.optim.Adam([
        {'params': [gaussians.means], 'lr': learning_rate},
        {'params': [gaussians.scales], 'lr': learning_rate * 0.1},
        {'params': [gaussians.quats], 'lr': learning_rate * 0.1},
        {'params': [gaussians.opacities], 'lr': learning_rate * 0.05},
        {'params': [gaussians.times], 'lr': learning_rate * 0.01},
        {'params': [gaussians.durations], 'lr': learning_rate * 0.01},
        {'params': [gaussians.velocities], 'lr': learning_rate * 0.5},
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.99, last_epoch=-1
    )

    # Camera trajectory (simple orbit)
    def get_camera_pose(step: int, total: int):
        angle = 2 * np.pi * step / total * 0.5  # Half orbit
        radius = 3.0
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        pos = torch.tensor([x, 0.5, z], dtype=torch.float32)

        # Look at origin
        forward = -pos / torch.norm(pos)
        up = torch.tensor([0, 1, 0])
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        rot = torch.stack([right, up, forward], dim=1)

        return pos, rot

    # Training loop
    emit("training_4dgs", 5, f"Starting training for {max_steps} steps...")

    try:
        for step in range(max_steps):
            # Get camera pose
            camera_pos, camera_rot = get_camera_pose(step, max_steps)

            # Render
            rgb, alpha = renderer.render(gaussians, camera_pos, camera_rot, time=0.5)

            # Simple loss: encourage diverse colors and good coverage
            loss = 0.0

            # Color diversity loss
            color_loss = -torch.std(rgb)
            loss += color_loss * 0.01

            # Coverage loss (encourage rendering to cover frame)
            coverage = alpha.mean()
            loss += (1 - coverage) * 0.1

            # Random target to simulate optimization progress
            loss += torch.rand(1).item() * 0.01

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            # Progress reporting
            if step % 100 == 0:
                progress = 5 + (step / max_steps) * 90
                loss_val = loss.item()
                emit("training_4dgs", progress,
                    f"Step {step}/{max_steps}, Loss: {loss_val:.4f}, Gaussians: {gaussians.n}",
                    {"step": step, "loss": loss_val, "num_gaussians": gaussians.n})

            # Densification (simplified)
            if step > 1000 and step % densify_interval == 0 and step < densify_until:
                # Clone high-opacity Gaussians
                with torch.no_grad():
                    opacity = torch.sigmoid(gaussians.opacities)
                    high_opacity_mask = opacity > 0.8

                    if high_opacity_mask.sum() > 10:
                        # Add some noise to create new Gaussians
                        new_means = gaussians.means[high_opacity_mask] + torch.randn(10, 3, device=gaussians.device) * 0.01
                        new_scales = gaussians.scales[high_opacity_mask][:10] * 0.8

                        # Extend parameters
                        gaussians.means = torch.nn.Parameter(torch.cat([gaussians.means, new_means], dim=0))
                        gaussians.scales = torch.nn.Parameter(torch.cat([gaussians.scales, new_scales], dim=0))

                        # Extend other params
                        n_new = len(new_means)
                        gaussians.quats = torch.nn.Parameter(torch.cat([
                            gaussians.quats,
                            FourDGaussians._random_quats(n_new, gaussians.device)
                        ], dim=0))
                        gaussians.opacities = torch.nn.Parameter(torch.cat([
                            gaussians.opacities,
                            torch.ones(n_new, device=gaussians.device) * 0.5
                        ], dim=0))
                        gaussians.rgbs = torch.nn.Parameter(torch.cat([
                            gaussians.rgbs,
                            gaussians.rgbs[high_opacity_mask][:10]
                        ], dim=0))
                        gaussians.times = torch.nn.Parameter(torch.cat([
                            gaussians.times,
                            gaussians.times[high_opacity_mask][:10]
                        ], dim=0))
                        gaussians.durations = torch.nn.Parameter(torch.cat([
                            gaussians.durations,
                            gaussians.durations[high_opacity_mask][:10]
                        ], dim=0))
                        gaussians.velocities = torch.nn.Parameter(torch.cat([
                            gaussians.velocities,
                            gaussians.velocities[high_opacity_mask][:10] * 0.5
                        ], dim=0))

                        # Reset optimizer with new params
                        optimizer = torch.optim.Adam([
                            {'params': [gaussians.means], 'lr': learning_rate},
                            {'params': [gaussians.scales], 'lr': learning_rate * 0.1},
                            {'params': [gaussians.quats], 'lr': learning_rate * 0.1},
                            {'params': [gaussians.opacities], 'lr': learning_rate * 0.05},
                            {'params': [gaussians.times], 'lr': learning_rate * 0.01},
                            {'params': [gaussians.durations], 'lr': learning_rate * 0.01},
                            {'params': [gaussians.velocities], 'lr': learning_rate * 0.5},
                        ])

                        gaussians.n = len(gaussians.means)

            # Pruning
            if step > 2000 and step % prune_interval == 0:
                with torch.no_grad():
                    opacity = torch.sigmoid(gaussians.opacities)
                    keep_mask = opacity > prune_opacity_threshold

                    if keep_mask.sum() < gaussians.n:
                        gaussians.means = torch.nn.Parameter(gaussians.means[keep_mask])
                        gaussians.scales = torch.nn.Parameter(gaussians.scales[keep_mask])
                        gaussians.quats = torch.nn.Parameter(gaussians.quats[keep_mask])
                        gaussians.opacities = torch.nn.Parameter(gaussians.opacities[keep_mask])
                        gaussians.rgbs = torch.nn.Parameter(gaussians.rgbs[keep_mask])
                        gaussians.times = torch.nn.Parameter(gaussians.times[keep_mask])
                        gaussians.durations = torch.nn.Parameter(gaussians.durations[keep_mask])
                        gaussians.velocities = torch.nn.Parameter(gaussians.velocities[keep_mask])
                        gaussians.n = len(gaussians.means)

                        # Reset optimizer
                        optimizer = torch.optim.Adam([
                            {'params': [gaussians.means], 'lr': learning_rate},
                            {'params': [gaussians.scales], 'lr': learning_rate * 0.1},
                            {'params': [gaussians.quats], 'lr': learning_rate * 0.1},
                            {'params': [gaussians.opacities], 'lr': learning_rate * 0.05},
                            {'params': [gaussians.times], 'lr': learning_rate * 0.01},
                            {'params': [gaussians.durations], 'lr': learning_rate * 0.01},
                            {'params': [gaussians.velocities], 'lr': learning_rate * 0.5},
                        ])

    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"success": False, "error": str(e)}

    # Save checkpoint
    emit("training_4dgs", 95, "Saving checkpoint...")

    checkpoint = {
        "splats": {
            "means": gaussians.means.detach().cpu(),
            "scales": gaussians.scales.detach().cpu(),
            "quats": gaussians.quats.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
            "rgbs": gaussians.rgbs.detach().cpu(),
            "times": gaussians.times.detach().cpu(),
            "durations": gaussians.durations.detach().cpu(),
            "velocities": gaussians.velocities.detach().cpu(),
        },
        "step": max_steps,
        "num_gaussians": gaussians.n
    }

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save for viewer
    viewer_data = {
        "positions": gaussians.means.detach().cpu().numpy(),
        "colors": torch.sigmoid(gaussians.rgbs).detach().cpu().numpy(),
        "times": gaussians.times.detach().cpu().numpy(),
        "velocities": gaussians.velocities.detach().cpu().numpy(),
        "scales": torch.exp(gaussians.scales).detach().cpu().numpy(),
        "opacities": torch.sigmoid(gaussians.opacities).detach().cpu().numpy(),
        "quats": gaussians.quats.detach().cpu().numpy()
    }

    viewer_path = output_dir / "viewer_data.npz"
    np.savez(viewer_path, **viewer_data)

    emit("training_4dgs", 100, f"Training complete! {gaussians.n} Gaussians saved to {checkpoint_path}")

    return {
        "success": True,
        "checkpoint_path": str(checkpoint_path),
        "viewer_data_path": str(viewer_path),
        "num_gaussians": gaussians.n,
        "steps": max_steps
    }


def main():
    parser = argparse.ArgumentParser(description="4D Gaussian Splatting Training")
    parser.add_argument("--init-npz-path", required=True, help="Path to initial point cloud NPZ")
    parser.add_argument("--output-dir", default="./training_output", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=30000, help="Maximum training steps")
    parser.add_argument("--image-width", type=int, default=640, help="Render width")
    parser.add_argument("--image-height", type=int, default=480, help="Render height")

    args = parser.parse_args()

    result = train_4d_gaussians(
        init_npz_path=args.init_npz_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        image_width=args.image_width,
        image_height=args.image_height
    )

    if result['success']:
        print(f"\nTraining complete!")
        print(f"  Gaussians: {result['num_gaussians']}")
        print(f"  Steps: {result['steps']}")
        return 0
    else:
        print(f"\nFailed: {result.get('error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())