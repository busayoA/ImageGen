import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import quaternion_to_rotation_matrix


class GaussianScene:
    """The Collection of 3D Gaussian primitives representing a scene"""


    def __init__(self, num_gaussians: int, device: str = 'cuda'):
        self.device = device
        self.num_gaussians = num_gaussians
        
        # Gaussian parameters (all learnable)
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3, device=device))
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4, device=device))
        self.scales = nn.Parameter(torch.randn(num_gaussians, 3, device=device))
        self.colors = nn.Parameter(torch.randn(num_gaussians, 3, device=device))
        self.opacities = nn.Parameter(torch.randn(num_gaussians, 1, device=device))
        
        # Normalize rotations
        self.rotations.data = F.normalize(self.rotations.data, dim=-1)
        
    def get_covariance_matrices(self) -> torch.Tensor:
        """Compute 3D covariance matrices from rotation and scale"""


        # Convert quaternions to rotation matrices
        R = quaternion_to_rotation_matrix(F.normalize(self.rotations, dim=-1))  # [N, 3, 3]
        
        # Create scaling matrices
        S = torch.diag_embed(torch.exp(self.scales))  # [N, 3, 3]
        
        # Compute covariance: Σ = R S S^T R^T
        RS = torch.bmm(R, S)  # [N, 3, 3]
        covariance = torch.bmm(RS, RS.transpose(-2, -1))  # [N, 3, 3]
        
        return covariance
    
    def get_colors(self) -> torch.Tensor:
        """Get RGB colors with sigmoid activation"""
        
        return torch.sigmoid(self.colors)
    
    def get_opacities(self) -> torch.Tensor:
        """Get opacities with sigmoid activation"""
        return torch.sigmoid(self.opacities)