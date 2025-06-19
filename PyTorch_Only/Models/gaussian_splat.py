import torch

from config import ProjectConfig
import torch.nn as nn
import torch.nn.functional as F

class GaussianSplat:
    """3D Gaussian Splat representation"""
    
    def __init__(self, positions: torch.Tensor, config: ProjectConfig):
        self.config = config
        
        # Gaussian parameters (all learnable)
        self.positions = nn.Parameter(positions)  # [N, 3]
        self.scales = nn.Parameter(torch.ones_like(positions) * 0.1)  # [N, 3]
        self.rotations = nn.Parameter(torch.zeros(len(positions), 4))  # [N, 4] quaternions
        self.colors = nn.Parameter(torch.rand(len(positions), 3))  # [N, 3] RGB
        self.opacities = nn.Parameter(torch.ones(len(positions), 1) * 0.5)  # [N, 1]
        
        # Initialize quaternions to identity
        self.rotations.data[:, 0] = 1.0
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """Compute 3D covariance matrices from scales and rotations"""
        # Convert quaternions to rotation matrices
        q = F.normalize(self.rotations, dim=1)
        R = self.quaternion_to_rotation_matrix(q)
        
        # Scale matrices
        S = torch.diag_embed(torch.abs(self.scales))
        
        # Covariance = R * S * S^T * R^T
        RS = torch.bmm(R, S)
        covariance = torch.bmm(RS, RS.transpose(-2, -1))
        
        return covariance
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices"""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(len(q), 3, 3, device=q.device)
        
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return R