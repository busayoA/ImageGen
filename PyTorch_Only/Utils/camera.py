from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from config import ProjectConfig


class CameraPoseGenerator:
    """Generate camera poses for multi-view synthesis"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        
    def generate_circular_poses(self, radius: float = 3.0, height: float = 0.0) -> torch.Tensor:
        """Generate camera poses in a circle around the object"""
        angles = torch.linspace(0, 2*np.pi, self.config.num_views, dtype=torch.float32)
        poses = []
        
        for angle in angles:
            # Camera position
            x = radius * torch.cos(angle)
            z = radius * torch.sin(angle)
            y = height
            
            # Look-at matrix (looking at origin)
            eye = torch.tensor([x, y, z])
            target = torch.tensor([0.0, 0.0, 0.0])
            up = torch.tensor([0.0, 1.0, 0.0])
            
            # Construct view matrix
            forward = F.normalize(target - eye, dim=0)
            right = F.normalize(torch.cross(forward, up), dim=0)
            up_new = torch.cross(right, forward)
            
            # Camera-to-world matrix
            pose = torch.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up_new
            pose[:3, 2] = -forward
            pose[:3, 3] = eye
            
            poses.append(pose)
            
        return torch.stack(poses)
    
    def poses_to_camera_params(self, poses: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert poses to camera parameters for conditioning"""
        # Extract rotation and translation
        rotations = poses[:, :3, :3]  # [N, 3, 3]
        translations = poses[:, :3, 3]  # [N, 3]
        
        # Convert to camera parameters (simplified)
        # In practice, you might want to use more sophisticated parameterization
        return {
            'rotations': rotations,
            'translations': translations,
            'fov': torch.full((len(poses),), self.config.fov)
        }