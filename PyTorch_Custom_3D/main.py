import torch
import numpy as np
import cv2
from GaussianViewsPipeline import GaussianViewsPipeline
from GaussianViewsEvaluator import GaussianViewsEvaluator
from utils import create_turntable_video, save_gaussian_scene

if __name__ == "__main__":
    # Initialize pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = GaussianViewsPipeline(device=device)
    
    # Generate scene
    text_prompt = "A red car parked beside a tree in a sunny field"
    scene = pipeline.generate_3d_scene(
        text_prompt=text_prompt,
        num_views=6,
        num_gaussians=50000,  # Reduced for faster testing - will increase later
        optimization_steps=1000  # Reduced for faster testing - will increase later
    )
    
    # Save scene
    save_gaussian_scene(scene, "generated_scene.npz")
    
    # Create turntable video
    create_turntable_video(pipeline, scene, "turntable.mp4", num_frames=60)
    
    # Render novel views
    novel_positions = [
        torch.tensor([2.0, 1.0, 2.0]),
        torch.tensor([-2.0, 0.5, 1.5]),
        torch.tensor([0.0, 2.0, 3.0])
    ]
    
    for i, pos in enumerate(novel_positions):
        image = pipeline.render_novel_view(scene, pos.to(device))
        
        # Save image
        try:
            import torchvision.utils as vutils
            vutils.save_image(image, f"novel_view_{i}.png")
        except ImportError:
            # Manual save if torchvision not available
            image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            cv2.imwrite(f"novel_view_{i}.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    print("Generation and rendering complete")
    
    # Evaluate the scene
    evaluator = GaussianViewsEvaluator()
    eval_poses = pipeline.generate_camera_poses(num_views=8, radius=2.5)
    for pose in eval_poses:
        pose.position = pose.position.to(device)
        pose.quaternion = pose.quaternion.to(device)
        pose.intrinsics = pose.intrinsics.to(device)
    
    results = evaluator.evaluate_scene(pipeline, scene, text_prompt, eval_poses)
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
