import torch
import torch.nn.functional as F
import math
from typing import List
from MultiViewDiffusionGenerator import MultiViewDiffusionGenerator
from GaussianScene import GaussianScene
from SemanticGaussianInitializer import SemanticGaussianInitializer
from GaussianViewsOptimizer import GaussianViewsOptimizer
from data_structures import CameraPose
from utils import rotation_matrix_to_quaternion


class GaussianViewsPipeline:
    """Main File with the complete GaussianViews pipeline"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.multiview_generator = MultiViewDiffusionGenerator()
        self.gaussian_initializer = SemanticGaussianInitializer()
        self.optimizer = GaussianViewsOptimizer()
        
    def generate_camera_poses(self, num_views: int = 4, radius: float = 3.0) -> List[CameraPose]:
        """Generate camera poses around the scene"""
        poses = []
        
        for i in range(num_views):
            # Circular camera trajectory
            angle = 2 * math.pi * i / num_views
            
            # Camera position
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            y = 0.0
            position = torch.tensor([x, y, z], dtype=torch.float32)
            
            # Look at origin - compute rotation
            forward = F.normalize(-position, dim=0)  # Look towards origin
            up = torch.tensor([0.0, 1.0, 0.0])
            right = F.normalize(torch.cross(forward, up), dim=0)
            up = torch.cross(right, forward)
            
            # Create rotation matrix and convert to quaternion
            R = torch.stack([right, up, -forward], dim=1)
            quaternion = rotation_matrix_to_quaternion(R.unsqueeze(0))[0]
            
            # Camera intrinsics (simplified implementation - BUILD ON THIS IN FUTURE)
            focal_length = 525.0
            cx, cy = 256.0, 256.0
            intrinsics = torch.tensor([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=torch.float32)
            
            pose = CameraPose(position=position, quaternion=quaternion, intrinsics=intrinsics)
            poses.append(pose)
            
        return poses
    
    def generate_3d_scene(self, 
                         text_prompt: str,
                         num_views: int = 4,
                         num_gaussians: int = 50000,
                         optimization_steps: int = 2000) -> GaussianScene:
        """Complete pipeline to generate 3D scene from text"""
        
        print(f"Generating 3D scene for: '{text_prompt}'")
        
        # Step 1: Generate camera poses
        print("Step 1: Generating camera poses...")
        poses = self.generate_camera_poses(num_views)
        
        # Step 2: Generate multi-view images
        print("Step 2: Generating multi-view images...")
        target_images = self.multiview_generator.generate_multiview_images(text_prompt, poses)
        
        # Convert to tensors and move to device
        target_images = [img.to(self.device) for img in target_images]
        for pose in poses:
            pose.position = pose.position.to(self.device)
            pose.quaternion = pose.quaternion.to(self.device)
            pose.intrinsics = pose.intrinsics.to(self.device)
        
        # Step 3: Initialize Gaussian scene
        print("Step 3: Initializing Gaussian scene...")
        scene = self.gaussian_initializer.initialize_gaussians(
            target_images, poses, text_prompt, num_gaussians
        )
        scene.positions = scene.positions.to(self.device)
        scene.rotations = scene.rotations.to(self.device)
        scene.scales = scene.scales.to(self.device)
        scene.colors = scene.colors.to(self.device)
        scene.opacities = scene.opacities.to(self.device)
        
        # Step 4: Optimize with spatial regularization
        print("Step 4: Optimizing Gaussian scene...")
        optimized_scene = self.optimizer.optimize(
            scene, target_images, poses, text_prompt, optimization_steps
        )
        
        print("Scene generation complete!")
        return optimized_scene
    
    def render_novel_view(self, 
                         scene: GaussianScene, 
                         camera_position: torch.Tensor,
                         look_at: torch.Tensor = None) -> torch.Tensor:
        """Render scene from novel viewpoint"""
        
        if look_at is None:
            look_at = torch.zeros(3, device=self.device)
        
        # Compute camera orientation
        forward = F.normalize(look_at - camera_position, dim=0)
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = F.normalize(torch.cross(forward, up), dim=0)
        up = torch.cross(right, forward)
        
        # Create rotation matrix and convert to quaternion
        R = torch.stack([right, up, -forward], dim=1)
        quaternion = rotation_matrix_to_quaternion(R.unsqueeze(0))[0]
        
        # Camera intrinsics
        focal_length = 525.0
        cx, cy = 256.0, 256.0
        intrinsics = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        # Create pose
        pose = CameraPose(
            position=camera_position,
            quaternion=quaternion,
            intrinsics=intrinsics
        )
        
        # Render
        with torch.no_grad():
            output = self.optimizer.renderer.render(scene, pose)
            
        return output['image']