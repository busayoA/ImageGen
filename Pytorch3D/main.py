import torch
import GaussianViewsPipeline as GaussianViewsPipeline


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = GaussianViewsPipeline(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate scene
    text_prompt = "A vintage red car parked beside a large oak tree in a sunny meadow"
    scene = pipeline.generate_3d_scene(
        text_prompt=text_prompt,
        num_views=6,
        num_gaussians=100000,
        optimization_steps=2000
    )
    
    # Save scene
    GaussianViewsPipeline.save_gaussian_scene(scene, "generated_scene.npz")
    
    # Create turntable video
    GaussianViewsPipeline.create_turntable_video(pipeline, scene, "turntable.mp4", num_frames=120)
    
    # Render novel views
    novel_positions = [
        torch.tensor([2.0, 1.0, 2.0]),
        torch.tensor([-2.0, 0.5, 1.5]),
        torch.tensor([0.0, 2.0, 3.0])
    ]
    
    for i, pos in enumerate(novel_positions):
        image = pipeline.render_novel_view(scene, pos.to(pipeline.device))
        
        # Save image
        import torchvision.utils as vutils
        vutils.save_image(image, f"novel_view_{i}.png")
    
    print("Generation and rendering complete!")