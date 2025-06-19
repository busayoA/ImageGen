import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from Models.renderer import GaussianRenderer
from Models.multiview_sd import MultiViewStableDiffusion
from Models.gaussian_splat import GaussianSplat
from Utils.camera import CameraPoseGenerator
from config import ProjectConfig
from pathlib import Path

class SpatialAIPipeline:
    """Main pipeline orchestrating all components"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.pose_generator = CameraPoseGenerator(config)
        self.diffusion_model = MultiViewStableDiffusion(config)
        self.renderer = GaussianRenderer(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_pipeline(self, prompt: str) -> Dict[str, any]:
        """Run the complete pipeline"""
        print(f"Starting 3D-aware generation for: '{prompt}'")
        
        # Step 1: Generate camera poses
        print("Generating camera poses...")
        poses = self.pose_generator.generate_circular_poses()
        camera_params = self.pose_generator.poses_to_camera_params(poses)
        
        # Step 2: Generate multi-view images
        print("Generating multi-view images...")
        images = self.diffusion_model.generate_multiview_images(prompt, poses)
        
        # Step 3: Initialize Gaussian splats from images
        print("Initializing 3D Gaussians...")
        initial_points = self.initialize_gaussians_from_images(images, poses)
        gaussians = GaussianSplat(initial_points, self.config)
        
        # Step 4: Optimize Gaussians
        print("Optimizing Gaussian splats...")
        optimized_gaussians = self.optimize_gaussians(
            gaussians, images, poses, camera_params
        )
        
        # Step 5: Save results
        results = {
            'prompt': prompt,
            'poses': poses,
            'images': images,
            'gaussians': optimized_gaussians
        }
        
        self.save_results(results)
        print(f"Pipeline complete! Results saved to {self.config.output_dir}")
        
        return results
    
    def initialize_gaussians_from_images(self, 
                                       images: List, 
                                       poses: torch.Tensor) -> torch.Tensor:
        """Initialize 3D points from multi-view images using simple heuristics"""
        # Simplified initialization - in practice use SfM or depth estimation
        num_points = self.config.num_gaussians
        
        # Create random points in a reasonable volume
        points = torch.randn(num_points, 3) * 2.0
        points[:, 1] -= 0.5  # Bias towards ground level
        
        return points
    
    def optimize_gaussians(self, 
                          gaussians: GaussianSplat,
                          target_images: List,
                          poses: torch.Tensor,
                          camera_params: Dict) -> GaussianSplat:
        """Optimize Gaussian parameters to match target images"""
        
        optimizer = torch.optim.Adam([
            gaussians.positions,
            gaussians.scales,
            gaussians.rotations,
            gaussians.colors,
            gaussians.opacities
        ], lr=self.config.learning_rate)
        
        # Simple intrinsics (in practice, calibrate properly)
        fx = fy = self.config.image_size / (2 * np.tan(np.radians(self.config.fov) / 2))
        cx = cy = self.config.image_size / 2
        intrinsics = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        for iteration in range(self.config.max_iterations):
            total_loss = 0
            
            for i, (target_image, pose) in enumerate(zip(target_images, poses)):
                # Render from current viewpoint
                rendered = self.renderer.render(gaussians, pose, intrinsics)
                
                # Convert target to tensor (simplified)
                target_tensor = torch.rand(3, self.config.image_size, self.config.image_size)
                
                # Compute loss
                loss = F.mse_loss(rendered, target_tensor)
                total_loss += loss
            
            # Backpropagate
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")
        
        return gaussians
    
    def save_results(self, results: Dict):
        """Save pipeline results"""
        import pickle
        
        # Save images
        for i, image in enumerate(results['images']):
            image.save(f"{self.config.output_dir}/view_{i:03d}.png")
        
        # Save Gaussian parameters
        gaussian_data = {
            'positions': results['gaussians'].positions.detach(),
            'scales': results['gaussians'].scales.detach(),
            'rotations': results['gaussians'].rotations.detach(),
            'colors': results['gaussians'].colors.detach(),
            'opacities': results['gaussians'].opacities.detach()
        }
        
        torch.save(gaussian_data, f"{self.config.output_dir}/gaussians.pt")
        
        # Save metadata
        metadata = {
            'prompt': results['prompt'],
            'config': self.config,
            'poses': results['poses']
        }
        
        with open(f"{self.config.output_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)




if __name__ == "__main__":
    # Initialize configuration
    config = ProjectConfig(
        image_size=512,
        num_views=8,
        num_gaussians=5000,
        max_iterations=500,
        output_dir="./3d_generation_output"
    )
    
    # Create and run pipeline
    pipeline = SpatialAIPipeline(config)
    
    # Example prompts to try
    prompts = [
        "a red vintage car in a forest clearing, photorealistic"
        # "a medieval castle on a hill, dramatic lighting",
        # "a modern chair in a minimalist room, studio lighting"
    ]
    
    for prompt in prompts:
        try:
            results = pipeline.run_pipeline(prompt)
            print(f"Successfully generated 3D scene for: {prompt}")
        except Exception as e:
            print(f"Error processing '{prompt}': {e}")
            continue

print("Project structure created! Next steps:")
print("1. Install required dependencies")
print("2. Adjust configuration parameters")
print("3. Implement missing PIL import: from PIL import Image")
print("4. Consider using pre-trained depth estimators like MiDaS")
print("5. Replace simplified components with production-ready versions")