from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
import time
import json
import os
import logging
from pathlib import Path
from typing import Optional
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import GaussianViewsPipeline
    from utils import save_gaussian_scene
    GAUSSIANVIEWS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GaussianViews not available: {e}")
    GAUSSIANVIEWS_AVAILABLE = False

app = FastAPI(
    title="GaussianViews Simple API",
    description="3D Scene Generation from Text",
    version="1.0.0"
)

# Global variables
pipeline = None
jobs = {}  # Simple in-memory job tracking

class GenerationRequest(BaseModel):
    prompt: str
    num_views: int = 4
    num_gaussians: int = 10000  # Reduced for CPU/small GPU
    optimization_steps: int = 200  # Reduced for faster generation
    image_size: int = 256  # Smaller images for faster processing

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: str
    prompt: str
    created_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None
    scene_path: Optional[str] = None
    video_path: Optional[str] = None

@app.on_event("startup")
async def startup():
    global pipeline
    
    logger.info("Starting GaussianViews API...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logger.info("CUDA not available, using CPU")
    
    if GAUSSIANVIEWS_AVAILABLE:
        try:
            logger.info("Initializing GaussianViews pipeline...")
            pipeline = GaussianViewsPipeline(device=device)
            logger.info("Pipeline ready!")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            pipeline = None
    else:
        logger.error("GaussianViews modules not available")
        pipeline = None

@app.get("/")
async def root():
    return {
        "message": "GaussianViews API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    cuda_available = torch.cuda.is_available()
    gpu_info = None
    
    if cuda_available:
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        }
    
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "gaussianViews_available": GAUSSIANVIEWS_AVAILABLE,
        "cuda_available": cuda_available,
        "gpu_info": gpu_info,
        "active_jobs": len(jobs),
        "torch_version": torch.__version__
    }

@app.get("/system-info")
async def system_info():
    """Get detailed system information"""
    import psutil
    
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "memory_available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
        "disk_usage": f"{psutil.disk_usage('/').free / (1024**3):.1f} GB free",
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__
    }

@app.post("/generate", response_model=JobResponse)
async def generate_scene(request: GenerationRequest, background_tasks: BackgroundTasks):
    if not GAUSSIANVIEWS_AVAILABLE:
        raise HTTPException(status_code=503, detail="GaussianViews not available")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    job_id = str(uuid.uuid4())[:8]  # Short ID for simplicity
    
    # Validate parameters
    if request.num_gaussians > 50000:
        raise HTTPException(status_code=400, detail="num_gaussians too high (max 50000)")
    
    if request.optimization_steps > 1000:
        raise HTTPException(status_code=400, detail="optimization_steps too high (max 1000)")
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": "Job queued for processing...",
        "prompt": request.prompt,
        "created_at": time.time(),
        "completed_at": None,
        "error": None,
        "scene_path": None,
        "video_path": None
    }
    
    # Start generation in background
    background_tasks.add_task(run_generation, job_id, request)
    
    logger.info(f"Started job {job_id} for prompt: {request.prompt}")
    
    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Generation queued for: {request.prompt[:50]}..."
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs[job_id])

@app.get("/jobs")
async def list_jobs():
    """List all jobs with their status"""
    return {
        "total_jobs": len(jobs),
        "jobs": [JobStatus(**job) for job in jobs.values()]
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Clean up files
    try:
        if job.get("scene_path") and os.path.exists(job["scene_path"]):
            os.remove(job["scene_path"])
        if job.get("video_path") and os.path.exists(job["video_path"]):
            os.remove(job["video_path"])
    except Exception as e:
        logger.warning(f"Failed to clean up files for job {job_id}: {e}")
    
    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}

def run_generation(job_id: str, request: GenerationRequest):
    """Run the actual generation process"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = "Initializing generation..."
        
        logger.info(f"Job {job_id}: Starting generation for '{request.prompt}'")
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        jobs[job_id]["progress"] = "Generating multi-view images..."
        
        # Run the generation with error handling
        try:
            scene = pipeline.generate_3d_scene(
                text_prompt=request.prompt,
                num_views=request.num_views,
                num_gaussians=request.num_gaussians,
                optimization_steps=request.optimization_steps
            )
        except Exception as e:
            logger.error(f"Job {job_id}: Generation failed: {e}")
            raise e
        
        jobs[job_id]["progress"] = "Saving 3D scene..."
        
        # Save the scene
        scene_path = output_dir / f"{job_id}_scene.npz"
        save_gaussian_scene(scene, str(scene_path))
        jobs[job_id]["scene_path"] = str(scene_path)
        
        logger.info(f"Job {job_id}: Scene saved to {scene_path}")
        
        # Try to create turntable video (optional)
        jobs[job_id]["progress"] = "Creating turntable video..."
        try:
            from utils import create_turntable_video
            video_path = output_dir / f"{job_id}_turntable.mp4"
            create_turntable_video(pipeline, scene, str(video_path), num_frames=24)
            jobs[job_id]["video_path"] = str(video_path)
            logger.info(f"Job {job_id}: Video saved to {video_path}")
        except Exception as e:
            logger.warning(f"Job {job_id}: Video creation failed: {e}")
            # Don't fail the whole job if video creation fails
        
        # Mark as completed
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Generation completed successfully!"
        jobs[job_id]["completed_at"] = time.time()
        
        duration = jobs[job_id]["completed_at"] - jobs[job_id]["created_at"]
        logger.info(f"Job {job_id}: Completed in {duration:.1f} seconds")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id}: Failed with error: {error_msg}")
        
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = f"Generation failed: {error_msg}"
        jobs[job_id]["error"] = error_msg
        jobs[job_id]["completed_at"] = time.time()

@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download generated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if file_type == "scene":
        file_path = job.get("scene_path")
    elif file_type == "video":
        file_path = job.get("video_path")
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'scene' or 'video'")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type='application/octet-stream'
    )

if __name__ == "__main__":
    import uvicorn
    
    # Create output directory
    Path("outputs").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )