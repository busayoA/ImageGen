import torch
import torch.nn.functional as F
import math
from typing import List, Dict
from GaussianRenderer import GaussianRenderer
from SpatialRegularizer import SpatialRegularizer
from GaussianScene import GaussianScene
from data_structures import CameraPose


class GaussianViewsOptimizer:
    """My main optimization loop for GaussianViews"""
    
    def __init__(self, 
                 learning_rates: Dict[str, float] = None,
                 regularization_weights: Dict[str, float] = None):
        
        self.learning_rates = learning_rates or {
            'positions': 0.01,
            'rotations': 0.001, 
            'scales': 0.001,
            'colors': 0.005,
            'opacities': 0.005
        }
        
        self.regularization_weights = regularization_weights or {
            'spatial': 0.1,
            'semantic': 0.05,
            'opacity': 0.01
        }
        
        self.renderer = GaussianRenderer()
        self.regularizer = SpatialRegularizer()
        
    def setup_optimizers(self, scene: GaussianScene) -> Dict[str, torch.optim.Optimizer]:
        """Setup separate optimizers for different parameter groups"""
        
        optimizers = {
            'positions': torch.optim.Adam([scene.positions], lr=self.learning_rates['positions']),
            'rotations': torch.optim.Adam([scene.rotations], lr=self.learning_rates['rotations']),
            'scales': torch.optim.Adam([scene.scales], lr=self.learning_rates['scales']),
            'colors': torch.optim.Adam([scene.colors], lr=self.learning_rates['colors']),
            'opacities': torch.optim.Adam([scene.opacities], lr=self.learning_rates['opacities'])
        }
        
        return optimizers
    
    def compute_total_loss(self,
                          scene: GaussianScene,
                          target_images: List[torch.Tensor],
                          poses: List[CameraPose],
                          text_prompt: str) -> Dict[str, torch.Tensor]:
        """Compute total loss with all regularization terms"""
        
        # Render from all viewpoints
        rendered_outputs = []
        for pose in poses:
            output = self.renderer.render(scene, pose)
            rendered_outputs.append(output)
        
        # Photometric reconstruction loss
        photometric_loss = 0.0
        for i, target in enumerate(target_images):
            rendered = rendered_outputs[i]['image']
            photometric_loss += F.mse_loss(rendered, target)
        photometric_loss /= len(target_images)
        
        # Spatial regularization losses
        depth_maps = [output['depth'] for output in rendered_outputs]
        geometric_loss = self.regularizer.geometric_consistency_loss(depth_maps, poses)
        normal_loss = self.regularizer.normal_consistency_loss(depth_maps, poses)
        
        # Semantic coherence loss
        rendered_images = [output['image'] for output in rendered_outputs]
        semantic_loss = self.regularizer.semantic_coherence_loss(
            rendered_images, target_images, text_prompt
        )
        
        # Opacity regularization
        opacity_loss = self.regularizer.opacity_regularization_loss(scene.get_opacities())
        
        # Combine losses
        spatial_loss = geometric_loss + normal_loss
        
        total_loss = (photometric_loss + 
                     self.regularization_weights['spatial'] * spatial_loss +
                     self.regularization_weights['semantic'] * semantic_loss +
                     self.regularization_weights['opacity'] * opacity_loss)
        
        return {
            'total': total_loss,
            'photometric': photometric_loss,
            'spatial': spatial_loss,
            'semantic': semantic_loss,
            'opacity': opacity_loss
        }
    
    def adaptive_gaussian_management(self,
                                   scene: GaussianScene,
                                   rendered_outputs: List[Dict[str, torch.Tensor]],
                                   step: int) -> GaussianScene:
        """Adaptively split, clone, and prune Gaussians during optimization"""
        
        if step % 100 != 0:  # Only perform management every 100 steps
            return scene
            
        device = scene.positions.device
        
        # Compute screen-space coverage for each Gaussian
        screen_coverages = []
        for output in rendered_outputs:
            alpha_map = output['alpha']
            # Simplified coverage computation
            coverage = alpha_map.sum()
            screen_coverages.append(coverage)
        
        avg_coverage = torch.stack(screen_coverages).mean()
        
        # Thresholds for management
        split_threshold = 0.005
        clone_threshold = 0.002
        prune_threshold = 0.005
        
        positions = scene.positions.data
        rotations = scene.rotations.data
        scales = scene.scales.data
        colors = scene.colors.data
        opacities = scene.opacities.data
        
        # Find Gaussians to split (large scale)
        max_scale = torch.exp(scales).max(dim=-1)[0]
        split_mask = max_scale > split_threshold
        
        # Find Gaussians to clone (high gradient)
        position_grads = scene.positions.grad
        if position_grads is not None:
            grad_norm = torch.norm(position_grads, dim=-1)
            clone_mask = grad_norm > clone_threshold
        else:
            clone_mask = torch.zeros_like(split_mask)
        
        # Find Gaussians to prune (low opacity)
        prune_mask = torch.sigmoid(opacities.squeeze()) < prune_threshold
        
        # Perform splitting
        if split_mask.any():
            split_indices = torch.where(split_mask)[0]
            num_splits = len(split_indices)
            
            # Create new Gaussians by splitting
            new_positions = positions[split_indices] + torch.randn(num_splits, 3, device=device) * 0.1
            new_rotations = rotations[split_indices]
            new_scales = scales[split_indices] - math.log(2)  # Half the scale
            new_colors = colors[split_indices]
            new_opacities = opacities[split_indices]
            
            # Add to scene
            positions = torch.cat([positions, new_positions])
            rotations = torch.cat([rotations, new_rotations])
            scales = torch.cat([scales, new_scales])
            colors = torch.cat([colors, new_colors])
            opacities = torch.cat([opacities, new_opacities])
            
            # Halve the scale of original Gaussians
            scales[split_indices] = scales[split_indices] - math.log(2)
        
        # Perform cloning
        if clone_mask.any():
            clone_indices = torch.where(clone_mask)[0]
            
            # Clone Gaussians
            positions = torch.cat([positions, positions[clone_indices]])
            rotations = torch.cat([rotations, rotations[clone_indices]])
            scales = torch.cat([scales, scales[clone_indices]])
            colors = torch.cat([colors, colors[clone_indices]])
            opacities = torch.cat([opacities, opacities[clone_indices]])
        
        # Perform pruning (keep non-pruned Gaussians)
        if prune_mask.any():
            keep_mask = ~prune_mask
            positions = positions[keep_mask]
            rotations = rotations[keep_mask]
            scales = scales[keep_mask]
            colors = colors[keep_mask]
            opacities = opacities[keep_mask]
        
        # Create new scene with updated Gaussians
        new_scene = GaussianScene(positions.shape[0], device=device)
        new_scene.positions.data = positions
        new_scene.rotations.data = F.normalize(rotations, dim=-1)  # Ensure normalized
        new_scene.scales.data = scales
        new_scene.colors.data = colors
        new_scene.opacities.data = opacities
        
        return new_scene
    
    def optimize(self,
                scene: GaussianScene,
                target_images: List[torch.Tensor],
                poses: List[CameraPose],
                text_prompt: str,
                num_steps: int = 2000) -> GaussianScene:
        """Main optimization loop sequencee"""
        
        optimizers = self.setup_optimizers(scene)
        
        # Learning rate scheduling
        schedulers = {}
        for name, optimizer in optimizers.items():
            schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        
        for step in range(num_steps):
            # Zero gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            # Compute losses
            losses = self.compute_total_loss(scene, target_images, poses, text_prompt)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([scene.positions, scene.rotations, 
                                          scene.scales, scene.colors, scene.opacities], 
                                         max_norm=1.0)
            
            # Optimizer steps
            for optimizer in optimizers.values():
                optimizer.step()
            
            # Update learning rates
            if step % 100 == 0:
                for scheduler in schedulers.values():
                    scheduler.step()
            
            # Adaptive Gaussian management
            if step > 500:  # Start management after initial convergence
                rendered_outputs = []
                with torch.no_grad():
                    for pose in poses:
                        output = self.renderer.render(scene, pose)
                        rendered_outputs.append(output)
                
                scene = self.adaptive_gaussian_management(scene, rendered_outputs, step)
                optimizers = self.setup_optimizers(scene)  # Reset optimizers for new scene
            
            # Logging
            if step % 100 == 0:
                print(f"Step {step}: Total Loss = {losses['total'].item():.4f}, "
                      f"Photometric = {losses['photometric'].item():.4f}, "
                      f"Spatial = {losses['spatial'].item():.4f}, "
                      f"Semantic = {losses['semantic'].item():.4f}")
        
        return scene