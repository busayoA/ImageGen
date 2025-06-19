"""
3D-Aware Image Generation with Gaussian Splatting
Project Structure and Core Implementation
"""

from dataclasses import dataclass

# Core dependencies (install with pip)
# pip install diffusers transformers torch torchvision
# pip install opencv-python numpy scipy
# pip install trimesh open3d accelerate


# pytorch3d - conda

@dataclass
class ProjectConfig:
    """Configuration for the entire pipeline"""
    # Generation settings
    image_size: int = 512
    num_views: int = 6
    batch_size: int = 1
    
    # Gaussian splatting settings
    num_gaussians: int = 10000
    max_iterations: int = 1000
    learning_rate: float = 0.01
    
    # Camera settings
    fov: float = 60.0  # Field of view in degrees
    near: float = 0.1
    far: float = 100.0
    
    # Output settings
    output_dir: str = "./outputs"
    save_intermediate: bool = True