import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from diffusers import StableDiffusionPipeline, ControlNetModel

from data_structures import CameraPose

class MultiViewDiffusionGenerator(nn.Module):
    """Multi-view conditioned diffusion model for generating consistent viewpoints"""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1"):
        super().__init__()
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        )
        
        # Camera pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(7, 256),  # position (3) + quaternion (4)
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, 768)  # Match cross-attention dimension
        )
        
    def encode_camera_pose(self, pose: CameraPose) -> torch.Tensor:
        """Encode camera pose for conditioning"""
        pose_vec = torch.cat([pose.position, pose.quaternion])
        return self.pose_encoder(pose_vec)
    
    def generate_synthetic_depth(self, pose: CameraPose, image_size: Tuple[int, int]) -> torch.Tensor:
        """Generate synthetic depth map for bootstrapping"""
        H, W = image_size
        
        # Create depth map with perspective projection
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        
        # Simple depth model: central object + background
        center_mask = (x**2 + y**2) < 0.5
        depth = torch.ones_like(x) * 10.0  # Background depth
        depth[center_mask] = 5.0 + torch.randn_like(depth[center_mask]) * 0.5  # Object depth
        
        return depth.unsqueeze(0)  # [1, H, W]
    
    def generate_multiview_images(self, 
                                 text_prompt: str, 
                                 poses: List[CameraPose],
                                 image_size: Tuple[int, int] = (512, 512)) -> List[torch.Tensor]:
        """Generate multi-view images from text prompt and camera poses"""

        
        images = []
        
        for pose in poses:
            # Generate synthetic depth conditioning
            depth_map = self.generate_synthetic_depth(pose, image_size)
            
            # Generate image with ControlNet depth conditioning
            image = self.pipe(
                prompt=text_prompt,
                image=depth_map,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            images.append(image_tensor.permute(2, 0, 1))  # [3, H, W]
            
        return images