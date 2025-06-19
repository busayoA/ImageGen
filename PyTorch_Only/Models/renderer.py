import torch
from Models.gaussian_splat import GaussianSplat
from config import ProjectConfig

class GaussianRenderer:
    """Differentiable Gaussian Splatting Renderer"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
    
    def render(self, 
               gaussians: GaussianSplat, 
               camera_pose: torch.Tensor,
               intrinsics: torch.Tensor) -> torch.Tensor:
        """Render Gaussians from given camera viewpoint"""
        
        # Transform to camera space
        world_to_cam = torch.inverse(camera_pose)
        positions_cam = self.transform_points(gaussians.positions, world_to_cam)
        
        # Project to screen space
        positions_screen = self.project_points(positions_cam, intrinsics)
        
        # Compute 2D covariance matrices
        covariance_3d = gaussians.get_covariance_matrices()
        covariance_2d = self.project_covariance(covariance_3d, world_to_cam, intrinsics)
        
        # Alpha blending
        image = self.alpha_blend(
            positions_screen,
            covariance_2d,
            gaussians.colors,
            gaussians.opacities
        )
        
        return image
    
    def transform_points(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Transform 3D points by 4x4 matrix"""
        points_h = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        return torch.mm(points_h, transform.T)[:, :3]
    
    def project_points(self, points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """Project camera space points to screen space"""
        # Simple perspective projection
        z = points_cam[:, 2:3]
        xy = points_cam[:, :2] / (z + 1e-8)
        
        # Apply intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        screen_x = xy[:, 0] * fx + cx
        screen_y = xy[:, 1] * fy + cy
        
        return torch.stack([screen_x, screen_y, z.squeeze()], dim=1)
    
    def project_covariance(self, 
                          cov_3d: torch.Tensor, 
                          world_to_cam: torch.Tensor,
                          intrinsics: torch.Tensor) -> torch.Tensor:
        """Project 3D covariance to 2D screen space"""
        # Simplified covariance projection
        # In practice, you'd want the full jacobian computation
        R = world_to_cam[:3, :3]
        cov_cam = torch.bmm(torch.bmm(R.unsqueeze(0).expand(len(cov_3d), -1, -1), 
                                     cov_3d), 
                           R.T.unsqueeze(0).expand(len(cov_3d), -1, -1))
        
        # Project to 2D (simplified)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        J = torch.tensor([[fx, 0], [0, fy]], device=cov_3d.device)
        
        cov_2d = torch.bmm(torch.bmm(J.unsqueeze(0).expand(len(cov_3d), -1, -1),
                                    cov_cam[:, :2, :2]),
                          J.T.unsqueeze(0).expand(len(cov_3d), -1, -1))
        
        return cov_2d
    
    def alpha_blend(self, 
                   positions: torch.Tensor,
                   covariances: torch.Tensor,
                   colors: torch.Tensor,
                   opacities: torch.Tensor) -> torch.Tensor:
        """Alpha blend Gaussians to create final image"""
        # Simplified rasterization - in practice use optimized CUDA kernels
        h, w = self.config.image_size, self.config.image_size
        image = torch.zeros(3, h, w, device=positions.device)
        alpha_acc = torch.zeros(h, w, device=positions.device)
        
        # Sort by depth
        _, indices = torch.sort(positions[:, 2])
        
        for idx in indices:
            pos = positions[idx]
            cov = covariances[idx]
            color = colors[idx]
            opacity = torch.sigmoid(opacities[idx])
            
            # Create 2D Gaussian
            self.splat_gaussian(image, alpha_acc, pos[:2], cov, color, opacity)
        
        return image
    
    def splat_gaussian(self, image, alpha_acc, center, cov, color, opacity):
        """Splat a single Gaussian onto the image"""
        # Simplified Gaussian splatting
        h, w = image.shape[1], image.shape[2]
        
        # Compute bounding box
        eigenvals, _ = torch.linalg.eigh(cov)
        radius = 3 * torch.sqrt(torch.max(eigenvals))
        
        x_min = max(0, int(center[0] - radius))
        x_max = min(w, int(center[0] + radius) + 1)
        y_min = max(0, int(center[1] - radius))
        y_max = min(h, int(center[1] + radius) + 1)
        
        # Sample Gaussian within bounding box
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                offset = torch.tensor([x - center[0], y - center[1]], device=center.device)
                
                # Compute Gaussian weight
                cov_inv = torch.inverse(cov + torch.eye(2, device=cov.device) * 1e-4)
                weight = torch.exp(-0.5 * torch.dot(offset, torch.mv(cov_inv, offset)))
                alpha = opacity * weight
                
                # Alpha blending
                current_alpha = alpha_acc[y, x]
                blend_alpha = alpha * (1 - current_alpha)
                
                image[:, y, x] += blend_alpha * color
                alpha_acc[y, x] += blend_alpha