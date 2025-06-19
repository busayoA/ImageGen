import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import CLIPModel, CLIPProcessor

from utils import quaternion_to_rotation_matrix
from data_structures import CameraPose

class SpatialRegularizer(nn.Module):
    """Spatial regularization for consistent 3D geometry"""
    
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") #this function call occurs too many times, can be standardised later
        
    def geometric_consistency_loss(self, 
                                  depth_maps: List[torch.Tensor],
                                  poses: List[CameraPose]) -> torch.Tensor:
        """Enforce geometric consistency across viewpoints"""
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(depth_maps)):
            for j in range(i+1, len(depth_maps)):
                # Compute depth gradients
                grad_i = self.compute_depth_gradients(depth_maps[i])
                grad_j = self.compute_depth_gradients(depth_maps[j])
                
                # Transform gradients between views (simplified)
                transformed_grad = self.transform_gradients(grad_j, poses[j], poses[i])
                
                # Consistency loss
                loss = F.mse_loss(grad_i, transformed_grad)
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def compute_depth_gradients(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradients of depth map"""
        
        grad_x = torch.diff(depth_map, dim=-1, prepend=depth_map[..., :1])
        grad_y = torch.diff(depth_map, dim=-2, prepend=depth_map[..., :1, :])
        return torch.stack([grad_x, grad_y], dim=0)
    
    def transform_gradients(self, 
                          gradients: torch.Tensor,
                          pose_from: CameraPose, 
                          pose_to: CameraPose) -> torch.Tensor:
        """Transform gradients between camera views"""

        # Compute relative transformation
        R_from = quaternion_to_rotation_matrix(pose_from.quaternion.unsqueeze(0))[0]
        R_to = quaternion_to_rotation_matrix(pose_to.quaternion.unsqueeze(0))[0]
        R_rel = torch.mm(R_to, R_from.T)
        
        # Apply transformation to gradients (simplified)
        return gradients  # For simplicity, return as-is
    
    def normal_consistency_loss(self, 
                              depth_maps: List[torch.Tensor],
                              poses: List[CameraPose]) -> torch.Tensor:
        """Enforce consistent surface normals across views"""
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(depth_maps)):
            for j in range(i+1, len(depth_maps)):
                # Compute surface normals from depth
                normals_i = self.depth_to_normals(depth_maps[i], poses[i])
                normals_j = self.depth_to_normals(depth_maps[j], poses[j])
                
                # Transform normals to common coordinate system
                normals_j_transformed = self.transform_normals(normals_j, poses[j], poses[i])
                
                # Consistency loss
                loss = F.cosine_embedding_loss(
                    normals_i.view(-1, 3), 
                    normals_j_transformed.view(-1, 3),
                    torch.ones(normals_i.numel() // 3, device=normals_i.device)
                )
                total_loss += loss
                num_pairs += 1
                
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def depth_to_normals(self, depth_map: torch.Tensor, pose: CameraPose) -> torch.Tensor:
        """Compute surface normals from depth map"""
        # Compute gradients
        grad_x = torch.diff(depth_map, dim=-1, prepend=depth_map[..., :1])
        grad_y = torch.diff(depth_map, dim=-2, prepend=depth_map[..., :1, :])
        
        # Convert to 3D normals (simplified)
        normals = torch.stack([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=0)
        normals = F.normalize(normals, dim=0)
        
        return normals.permute(1, 2, 0)  # [H, W, 3]
    
    def transform_normals(self, 
                         normals: torch.Tensor,
                         pose_from: CameraPose,
                         pose_to: CameraPose) -> torch.Tensor:
        """Transform normals between coordinate systems"""
        # Compute relative rotation
        R_from = quaternion_to_rotation_matrix(pose_from.quaternion.unsqueeze(0))[0]
        R_to = quaternion_to_rotation_matrix(pose_to.quaternion.unsqueeze(0))[0]
        R_rel = torch.mm(R_to, R_from.T)
        
        # Transform normals
        H, W, _ = normals.shape
        normals_flat = normals.view(-1, 3)
        normals_transformed = torch.mm(normals_flat, R_rel.T)
        
        return normals_transformed.view(H, W, 3)
    
    def semantic_coherence_loss(self, 
                              rendered_images: List[torch.Tensor],
                              target_images: List[torch.Tensor],
                              text_prompt: str) -> torch.Tensor:
        """Enforce semantic consistency using CLIP"""
        total_loss = 0.0
        
        for rendered, target in zip(rendered_images, target_images):
            # Compute CLIP features
            rendered_features = self.get_clip_features(rendered)
            target_features = self.get_clip_features(target)
            
            # Semantic consistency loss
            loss = F.mse_loss(rendered_features, target_features)
            total_loss += loss
            
        return total_loss / len(rendered_images)
    
    def get_clip_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract CLIP features from image"""
        # Convert to PIL format
        image_pil = torch.clamp(image * 255, 0, 255).byte().permute(1, 2, 0).numpy()
        
        # Process with CLIP
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=image_pil, return_tensors="pt")
        features = self.clip_model.get_image_features(**inputs)
        
        return features
    
    def opacity_regularization_loss(self, opacities: torch.Tensor) -> torch.Tensor:
        """Encourage binary opacity values to prevent floating artifacts"""
        # Binary regularization: encourage 0 or 1 values
        binary_loss = (opacities * (1 - opacities)).mean()
        return binary_loss