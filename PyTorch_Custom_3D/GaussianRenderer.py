import torch
import torch.nn as nn
from typing import Tuple, Dict

from GaussianScene import GaussianScene
from data_structures import CameraPose
from utils import quaternion_to_rotation_matrix

class GaussianRenderer(nn.Module):
    """Differentiable Gaussian splatting renderer"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.image_size = image_size
        
    def render(self, 
              scene: GaussianScene, 
              pose: CameraPose,
              background_color: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Render scene from given camera pose"""
        
        if background_color is None:
            background_color = torch.zeros(3, device=scene.positions.device)
            
        H, W = self.image_size
        
        # Transform Gaussians to camera space
        gaussians_cam = self.transform_gaussians_to_camera(scene, pose)
        
        # Project to screen space
        screen_coords, depths = self.project_gaussians(gaussians_cam, pose.intrinsics)
        
        # Sort by depth (back to front)
        depth_indices = torch.argsort(depths, descending=True)
        
        # Rasterize Gaussians
        rendered_image, depth_map, alpha_map = self.rasterize_gaussians(
            gaussians_cam, screen_coords, depth_indices, background_color
        )
        
        return {
            'image': rendered_image,
            'depth': depth_map, 
            'alpha': alpha_map
        }
    
    def transform_gaussians_to_camera(self, 
                                    scene: GaussianScene,
                                    pose: CameraPose) -> Dict[str, torch.Tensor]:
        """Transform Gaussians from world to camera coordinates"""
        
        # Camera transformation
        R = quaternion_to_rotation_matrix(pose.quaternion.unsqueeze(0))[0]
        t = pose.position
        
        # Transform positions
        positions_cam = torch.mm(scene.positions - t.unsqueeze(0), R.T)
        
        # Transform covariances
        covariances_world = scene.get_covariance_matrices()
        covariances_cam = torch.bmm(torch.bmm(R.unsqueeze(0).expand(scene.num_gaussians, -1, -1), 
                                             covariances_world), 
                                   R.T.unsqueeze(0).expand(scene.num_gaussians, -1, -1))
        
        return {
            'positions': positions_cam,
            'covariances': covariances_cam,
            'colors': scene.get_colors(),
            'opacities': scene.get_opacities()
        }
    
    def project_gaussians(self, 
                         gaussians_cam: Dict[str, torch.Tensor],
                         intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D Gaussians to screen space"""
        
        positions = gaussians_cam['positions']  # [N, 3]
        
        # Perspective projection
        projected = torch.mm(positions, intrinsics.T)  # [N, 3]
        screen_coords = projected[:, :2] / projected[:, 2:3]  # [N, 2]
        depths = projected[:, 2]  # [N]
        
        return screen_coords, depths
    
    def rasterize_gaussians(self,
                          gaussians_cam: Dict[str, torch.Tensor],
                          screen_coords: torch.Tensor, 
                          depth_indices: torch.Tensor,
                          background_color: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rasterize sorted Gaussians to image"""
        
        H, W = self.image_size
        device = screen_coords.device
        
        # Initialize output buffers
        final_color = background_color.view(3, 1, 1).expand(3, H, W).clone()
        final_depth = torch.zeros(H, W, device=device)
        final_alpha = torch.zeros(H, W, device=device)
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        
        # Render each Gaussian (back to front)
        transmittance = torch.ones(H, W, device=device)
        
        for idx in depth_indices:
            if transmittance.max() < 0.001:  # Early termination
                break
                
            # Get Gaussian parameters
            center = screen_coords[idx]  # [2]
            color = gaussians_cam['colors'][idx]  # [3]
            opacity = gaussians_cam['opacities'][idx]  # [1]
            depth = gaussians_cam['positions'][idx, 2]  # scalar
            
            # Compute 2D covariance (simplified - will project 3D covariance in furture)
            cov_2d = torch.eye(2, device=device) * 25.0  # Fixed covariance for simplicity
            
            # Compute Gaussian weights
            diff = pixel_coords - center.view(1, 1, 2)  # [H, W, 2]
            
            # Compute Mahalanobis distance
            cov_inv = torch.inverse(cov_2d)
            mahal_dist = torch.sum(diff * torch.mm(diff.view(-1, 2), cov_inv).view(H, W, 2), dim=-1)
            
            # Gaussian weight
            weight = torch.exp(-0.5 * mahal_dist)  # [H, W]
            
            # Alpha compositing
            alpha = weight * opacity.item()
            alpha = torch.clamp(alpha, 0, 1)
            
            # Update color
            final_color += transmittance.unsqueeze(0) * alpha.unsqueeze(0) * color.view(3, 1, 1)
            
            # Update depth (weighted average)
            final_depth += transmittance * alpha * depth.item()
            
            # Update alpha and transmittance
            final_alpha += transmittance * alpha
            transmittance *= (1 - alpha)
        
        return final_color, final_depth, final_alpha