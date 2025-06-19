import torch
import numpy as np
from typing import List, Dict
from transformers import CLIPModel, CLIPProcessor
from GaussianViewsPipeline import GaussianViewsPipeline
from GaussianScene import GaussianScene
from data_structures import CameraPose

class GaussianViewsEvaluator:
    """Evaluation for results"""
    
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def compute_multiview_consistency(self, images: List[torch.Tensor]) -> float:
        """Compute multi-view LPIPS consistency"""

        try:
            import lpips
            loss_fn = lpips.LPIPS(net='alex')
        except ImportError:
            print("LPIPS not found. Install with: pip install lpips")
            return 0.0
        
        total_loss = 0.0
        num_pairs = 0
        
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                loss = loss_fn(images[i].unsqueeze(0), images[j].unsqueeze(0))
                total_loss += loss.item()
                num_pairs += 1
                
        return total_loss / num_pairs if num_pairs > 0 else 0.0
    
    def compute_clip_similarity(self, image: torch.Tensor, text_prompt: str) -> float:
        """Compute CLIP similarity between image and text"""
        

        # Prepare inputs
        image_pil = (image.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        inputs = self.clip_processor(
            text=[text_prompt], 
            images=image_pil, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get features
        outputs = self.clip_model(**inputs)
        similarity = outputs.logits_per_image.item()
        
        return similarity
    
    def evaluate_scene(self, 
                      pipeline: GaussianViewsPipeline,
                      scene: GaussianScene,
                      text_prompt: str,
                      evaluation_poses: List[CameraPose]) -> Dict[str, float]:
        """Comprehensive scene evaluation"""
        

        
        # Render from evaluation poses
        rendered_images = []
        rendered_depths = []
        
        with torch.no_grad():
            for pose in evaluation_poses:
                output = pipeline.optimizer.renderer.render(scene, pose)
                rendered_images.append(output['image'])
                rendered_depths.append(output['depth'])
        
        # Compute metrics
        results = {}
        
        # Multi-view consistency
        results['multiview_lpips'] = self.compute_multiview_consistency(rendered_images)
        
        # CLIP similarity
        clip_scores = []
        for image in rendered_images:
            score = self.compute_clip_similarity(image, text_prompt)
            clip_scores.append(score)
        results['clip_score'] = np.mean(clip_scores)
        results['clip_std'] = np.std(clip_scores)
        
        # Scene statistics
        results['num_gaussians'] = scene.num_gaussians
        results['opacity_mean'] = scene.get_opacities().mean().item()
        results['opacity_std'] = scene.get_opacities().std().item()
        
        return results