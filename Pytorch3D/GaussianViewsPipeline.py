import math
from typing import List
import numpy as np
import torch.nn.functional as F
import torch
import GaussianViewsOptimizer, MultiViewDiffusionGenerator, SemanticGaussianInitializer
from Data_Structures.CameraPose import CameraPose
from Pytorch3D.GaussianScene import GaussianScene


class GaussianViewsPipeline:
    """Complete GaussianViews pipeline"""
    
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
            
            # Look at origin
            forward = -position / torch.norm(position)
            up = torch.tensor([0.0, 1.0, 0.0])
            right = torch.cross(forward, up)
            up = torch.cross(right, forward)
            
            # Rotation matrix to quaternion
            R = torch.stack([right, up, -forward], dim=1)
            quaternion = self.rotation_matrix_to_quaternion(R)
            
            # Camera intrinsics (simplified)
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
    
    def rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion"""
        # Simplified conversion (use proper implementation in practice)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S
                
        return torch.tensor([w, x, y, z], dtype=torch.float32)
    
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
        quaternion = self.rotation_matrix_to_quaternion(R)
        
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

# Utility functions for visualization and export

def save_gaussian_scene(scene: GaussianScene, filepath: str):
    """Save Gaussian scene to file"""
    data = {
        'positions': scene.positions.detach().cpu().numpy(),
        'rotations': scene.rotations.detach().cpu().numpy(),
        'scales': scene.scales.detach().cpu().numpy(),
        'colors': scene.colors.detach().cpu().numpy(),
        'opacities': scene.opacities.detach().cpu().numpy(),
        'num_gaussians': scene.num_gaussians
    }
    np.savez(filepath, **data)

def load_gaussian_scene(filepath: str, device: str = 'cuda') -> GaussianScene:
    """Load Gaussian scene from file"""
    data = np.load(filepath)
    
    scene = GaussianScene(data['num_gaussians'], device=device)
    scene.positions.data = torch.from_numpy(data['positions']).to(device)
    scene.rotations.data = torch.from_numpy(data['rotations']).to(device)
    scene.scales.data = torch.from_numpy(data['scales']).to(device)
    scene.colors.data = torch.from_numpy(data['colors']).to(device)
    scene.opacities.data = torch.from_numpy(data['opacities']).to(device)
    
    return scene

def create_turntable_video(pipeline: GaussianViewsPipeline,
                          scene: GaussianScene,
                          output_path: str,
                          num_frames: int = 60,
                          radius: float = 3.0):
    """Create turntable video of the scene"""
    import imageio # type: ignore
    
    frames = []
    
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        
        # Camera position
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = 0.0
        camera_pos = torch.tensor([x, y, z], device=pipeline.device)
        
        # Render frame
        image = pipeline.render_novel_view(scene, camera_pos)
        
        # Convert to numpy
        frame = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        frame = frame.transpose(1, 2, 0)  # CHW to HWC
        frames.append(frame)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=30)
