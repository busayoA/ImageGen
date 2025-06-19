import torch
import torch.nn.functional as F
import math
from typing import List, Tuple
from transformers import CLIPModel, CLIPProcessor

from Data_Structures.CameraPose import CameraPose
from Pytorch3D.GaussianScene import GaussianScene
from pytorch3d.transforms import quaternion_to_matrix # type: ignore

class SemanticGaussianInitializer:
    """Semantic-aware initialization of Gaussian primitives using CLIP guidance"""
    
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def compute_clip_similarity(self, 
                               image_patches: torch.Tensor, 
                               text_prompt: str) -> torch.Tensor:
        """Compute CLIP similarity between image patches and text"""
        # Process text
        text_inputs = self.clip_processor(text=[text_prompt], return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Process image patches
        similarities = []
        for patch in image_patches:
            patch_pil = torch.clamp(patch * 255, 0, 255).byte().permute(1, 2, 0).numpy()
            inputs = self.clip_processor(images=patch_pil, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(text_features, image_features)
            similarities.append(similarity)
            
        return torch.stack(similarities)
    
    def structure_from_motion(self, 
                             images: List[torch.Tensor], 
                             poses: List[CameraPose]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract 3D point cloud from multi-view images using SfM"""
        # Simplified SfM - in practice use COLMAP or similar
        points_3d = []
        colors = []
        
        H, W = images[0].shape[1], images[0].shape[2]
        
        # Sample points on a sphere for initial structure
        num_points = 10000
        phi = torch.rand(num_points) * 2 * math.pi
        theta = torch.rand(num_points) * math.pi
        r = 2.0 + torch.randn(num_points) * 0.5
        
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)  
        z = r * torch.cos(theta)
        
        points_3d = torch.stack([x, y, z], dim=1)  # [N, 3]
        
        # Sample colors from first image
        u = torch.randint(0, W, (num_points,))
        v = torch.randint(0, H, (num_points,))
        colors = images[0][:, v, u].T  # [N, 3]
        
        return points_3d, colors
    
    def estimate_semantic_density(self, 
                                 points_3d: torch.Tensor,
                                 images: List[torch.Tensor],
                                 poses: List[CameraPose],
                                 text_prompt: str) -> torch.Tensor:
        """Estimate required Gaussian density based on semantic importance"""
        densities = torch.zeros(points_3d.shape[0])
        
        for i, (image, pose) in enumerate(zip(images, poses)):
            # Project 3D points to image plane
            projected_points = self.project_points(points_3d, pose)
            
            # Extract patches around projected points
            patch_size = 32
            patches = []
            valid_indices = []
            
            H, W = image.shape[1], image.shape[2]
            
            for j, (u, v) in enumerate(projected_points):
                u, v = int(u), int(v)
                if (patch_size//2 <= u < W - patch_size//2 and 
                    patch_size//2 <= v < H - patch_size//2):
                    
                    patch = image[:, v-patch_size//2:v+patch_size//2, 
                                    u-patch_size//2:u+patch_size//2]
                    patches.append(patch)
                    valid_indices.append(j)
            
            if patches:
                patches_tensor = torch.stack(patches)
                similarities = self.compute_clip_similarity(patches_tensor, text_prompt)
                
                for k, idx in enumerate(valid_indices):
                    densities[idx] += similarities[k].item()
        
        return densities / len(images)  # Average across views
    
    def project_points(self, points_3d: torch.Tensor, pose: CameraPose) -> torch.Tensor:
        """Project 3D points to 2D image coordinates"""
        # Transform to camera coordinates
        R = quaternion_to_matrix(pose.quaternion.unsqueeze(0))[0]
        t = pose.position
        
        points_cam = torch.mm(points_3d - t.unsqueeze(0), R.T)
        
        # Project to image plane
        K = pose.intrinsics
        points_2d_homo = torch.mm(points_cam, K.T)
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        return points_2d
    
    def initialize_gaussians(self, 
                           images: List[torch.Tensor],
                           poses: List[CameraPose], 
                           text_prompt: str,
                           num_gaussians: int = 50000) -> GaussianScene:
        """Initialize Gaussian scene with semantic guidance"""
        
        # Extract 3D structure
        points_3d, colors = self.structure_from_motion(images, poses)
        
        # Estimate semantic density
        semantic_densities = self.estimate_semantic_density(points_3d, images, poses, text_prompt)
        
        # Sample Gaussian positions based on density
        probabilities = F.softmax(semantic_densities * 2.0, dim=0)  # Temperature scaling
        indices = torch.multinomial(probabilities, num_gaussians, replacement=True)
        
        # Initialize Gaussian scene
        scene = GaussianScene(num_gaussians)
        
        # Set positions from sampled points
        scene.positions.data = points_3d[indices].clone()
        
        # Set colors from corresponding image colors
        scene.colors.data = torch.logit(torch.clamp(colors[indices], 0.01, 0.99))
        
        # Initialize scales based on local point density
        knn_distances = self.compute_knn_distances(scene.positions.data, k=3)
        scene.scales.data = torch.log(knn_distances.unsqueeze(-1).expand(-1, 3))
        
        # Initialize rotations randomly (normalized)
        scene.rotations.data = F.normalize(torch.randn_like(scene.rotations.data), dim=-1)
        
        # Initialize opacities based on semantic importance
        semantic_importance = semantic_densities[indices]
        scene.opacities.data = torch.logit(torch.clamp(semantic_importance.unsqueeze(-1), 0.01, 0.99))
        
        return scene
    
    def compute_knn_distances(self, points: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Compute average distance to k nearest neighbors"""
        distances = torch.cdist(points, points)
        knn_distances, _ = torch.topk(distances, k+1, dim=-1, largest=False)
        return knn_distances[:, 1:].mean(dim=-1)  # Exclude self-distance