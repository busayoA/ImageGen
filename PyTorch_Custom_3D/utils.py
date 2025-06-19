import math
import numpy as np
import torch
import torch.nn.functional as F
import GaussianScene
import GaussianViewsPipeline as GaussianViewsPipeline


# Custom 3D transformations (because I can't use PyTorch3D)

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    My manual implementation of the quaternion_to_rotation_matrix function of PyTorch3D

    Convert quaternions to rotation matrices.
    Args:
        quaternions: [N, 4] tensor of quaternions (w, x, y, z)
    Returns:
        rotation_matrices: [N, 3, 3] tensor of rotation matrices
    """
    # Normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    
    w, x, y, z = quaternions.unbind(-1)
    
    # Compute rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Build rotation matrix
    rotation_matrices = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)
    
    return rotation_matrices

def rotation_matrix_to_quaternion(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    My manual implementation of the rotation_matrix_to_quaternion function of PyTorch3D

    Convert rotation matrices to quaternions.
    Args:
        rotation_matrices: [N, 3, 3] tensor of rotation matrices
    Returns:
        quaternions: [N, 4] tensor of quaternions (w, x, y, z)
    """
    batch_size = rotation_matrices.shape[0]
    device = rotation_matrices.device
    
    # Extract diagonal and off-diagonal elements
    m00 = rotation_matrices[:, 0, 0]
    m11 = rotation_matrices[:, 1, 1]
    m22 = rotation_matrices[:, 2, 2]
    
    m01 = rotation_matrices[:, 0, 1]
    m02 = rotation_matrices[:, 0, 2]
    m10 = rotation_matrices[:, 1, 0]
    m12 = rotation_matrices[:, 1, 2]
    m20 = rotation_matrices[:, 2, 0]
    m21 = rotation_matrices[:, 2, 1]
    
    # Compute quaternion components
    trace = m00 + m11 + m22
    
    # Initialize quaternion tensor
    quaternions = torch.zeros(batch_size, 4, device=device)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
        quaternions[mask1, 0] = 0.25 * s  # qw
        quaternions[mask1, 1] = (m21[mask1] - m12[mask1]) / s  # qx
        quaternions[mask1, 2] = (m02[mask1] - m20[mask1]) / s  # qy
        quaternions[mask1, 3] = (m10[mask1] - m01[mask1]) / s  # qz
    
    # Case 2: m00 > m11 and m00 > m22
    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    if mask2.any():
        s = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2  # s = 4 * qx
        quaternions[mask2, 0] = (m21[mask2] - m12[mask2]) / s  # qw
        quaternions[mask2, 1] = 0.25 * s  # qx
        quaternions[mask2, 2] = (m01[mask2] + m10[mask2]) / s  # qy
        quaternions[mask2, 3] = (m02[mask2] + m20[mask2]) / s  # qz
    
    # Case 3: m11 > m22
    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    if mask3.any():
        s = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2  # s = 4 * qy
        quaternions[mask3, 0] = (m02[mask3] - m20[mask3]) / s  # qw
        quaternions[mask3, 1] = (m01[mask3] + m10[mask3]) / s  # qx
        quaternions[mask3, 2] = 0.25 * s  # qy
        quaternions[mask3, 3] = (m12[mask3] + m21[mask3]) / s  # qz
    
    # Case 4: else
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2  # s = 4 * qz
        quaternions[mask4, 0] = (m10[mask4] - m01[mask4]) / s  # qw
        quaternions[mask4, 1] = (m02[mask4] + m20[mask4]) / s  # qx
        quaternions[mask4, 2] = (m12[mask4] + m21[mask4]) / s  # qy
        quaternions[mask4, 3] = 0.25 * s  # qz
    
    return quaternions

def euler_to_quaternion(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    A manual rotation_matrix_to_quaternion implementation based on the PyTorch3D library documentation

    Convert Euler angles (roll, pitch, yaw) to quaternions.
    Args:
        euler_angles: [N, 3] tensor of Euler angles in radians
    Returns:
        quaternions: [N, 4] tensor of quaternions (w, x, y, z)
    """
    roll, pitch, yaw = euler_angles.unbind(-1)
    
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([w, x, y, z], dim=-1)

def look_at_matrix(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Create a look-at transformation matrix.
    Args:
        eye: [3] camera position
        target: [3] target position
        up: [3] up vector
    Returns:
        transform_matrix: [4, 4] transformation matrix
    """
    forward = F.normalize(target - eye, dim=0)
    right = F.normalize(torch.cross(forward, up), dim=0)
    up_new = torch.cross(right, forward)
    
    # Create rotation matrix
    rotation = torch.stack([right, up_new, -forward], dim=1)
    
    # Create full transformation matrix
    transform_matrix = torch.eye(4, device=eye.device)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = eye
    
    return transform_matrix


def save_gaussian_scene(scene: GaussianScene, filepath: str):
    """Save Gaussian scene to a file"""
    data = {
        'positions': scene.positions.detach().cpu().numpy(),
        'rotations': scene.rotations.detach().cpu().numpy(),
        'scales': scene.scales.detach().cpu().numpy(),
        'colors': scene.colors.detach().cpu().numpy(),
        'opacities': scene.opacities.detach().cpu().numpy(),
        'num_gaussians': scene.num_gaussians
    }
    np.savez(filepath, **data)

def load_gaussian_scene(filepath: str, device: str = 'cuda') -> GaussianScene:
    """Load a Gaussian scene from a givenfile"""
    data = np.load(filepath)
    
    scene = GaussianScene(data['num_gaussians'], device=device)
    scene.positions.data = torch.from_numpy(data['positions']).to(device)
    scene.rotations.data = torch.from_numpy(data['rotations']).to(device)
    scene.scales.data = torch.from_numpy(data['scales']).to(device)
    scene.colors.data = torch.from_numpy(data['colors']).to(device)
    scene.opacities.data = torch.from_numpy(data['opacities']).to(device)
    
    return scene

def create_turntable_video(pipeline: GaussianViewsPipeline,
                          scene: GaussianScene,
                          output_path: str,
                          num_frames: int = 60,
                          radius: float = 3.0):
    """Create turntable video of the scene"""
    try:
        import imageio
    except ImportError:
        print("imageio not found. Install with: pip install imageio")
        return
    
    frames = []
    
    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        
        # Camera position
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        y = 0.0
        camera_pos = torch.tensor([x, y, z], device=pipeline.device)
        
        # Render frame
        image = pipeline.render_novel_view(scene, camera_pos)
        
        # Convert to numpy
        frame = (image.detach().cpu().numpy() * 255).astype(np.uint8)
        frame = frame.transpose(1, 2, 0)  # CHW to HWC
        frames.append(frame)
    
    # Save video
    imageio.mimsave(output_path, frames, fps=30)