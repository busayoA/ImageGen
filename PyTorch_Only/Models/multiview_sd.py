import numpy as np
import torch
from config import ProjectConfig
import cv2
from PIL import Image

class MultiViewStableDiffusion:
    """Stable Diffusion pipeline with multi-view generation"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.setup_pipeline()
        
    def setup_pipeline(self):
        """Initialize the Stable Diffusion pipeline with ControlNet"""
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        
        # Load ControlNet for depth conditioning
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16
        )
        
        # Load main pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
    
    def generate_depth_maps(self, poses: torch.Tensor) -> torch.Tensor:
        """Generate synthetic depth maps for conditioning"""
        # Simplified depth map generation
        # In practice, you'd want more sophisticated scene modeling
        depth_maps = []
        
        for pose in poses:
            # Create a simple depth map (sphere + ground plane)
            h, w = self.config.image_size, self.config.image_size
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing='ij'
            )
            
            # Sphere in center
            r = torch.sqrt(x**2 + y**2)
            sphere_depth = torch.where(r < 0.5, 0.8 - torch.sqrt(0.25 - r**2), 1.0)
            
            # Ground plane
            ground_depth = torch.where(y > 0.3, y * 0.5 + 0.8, sphere_depth)
            
            depth_maps.append(ground_depth)
            
        return torch.stack(depth_maps)
    
    def generate_multiview_images(self, 
                                 prompt: str, 
                                 poses: torch.Tensor,
                                 guidance_scale: float = 7.5,
                                 num_inference_steps: int = 20) -> torch.Tensor:
        """Generate images from multiple viewpoints"""
        
        # Generate depth maps for conditioning
        depth_maps = self.generate_depth_maps(poses)
        
        images = []
        for i, depth_map in enumerate(depth_maps):
            # Convert depth map to PIL Image for ControlNet
            depth_image = (depth_map.numpy() * 255).astype(np.uint8)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            depth_pil = Image.fromarray(depth_image)
            
            # Add camera pose information to prompt
            angle = i * (360 // self.config.num_views)
            view_prompt = f"{prompt}, view from {angle} degrees"
            
            # Generate image
            result = self.pipe(
                prompt=view_prompt,
                image=depth_pil,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator().manual_seed(42 + i)
            )
            
            images.append(result.images[0])
        
        return images