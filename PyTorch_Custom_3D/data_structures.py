from dataclasses import dataclass
import torch


@dataclass
class CameraPose:
    position: torch.Tensor  # [3]
    quaternion: torch.Tensor  # [4] 
    intrinsics: torch.Tensor  # [3, 3]

@dataclass
class GaussianPrimitive:
    position: torch.Tensor  # [3]
    rotation: torch.Tensor  # [4] quaternion
    scale: torch.Tensor     # [3]
    color: torch.Tensor     # [3]
    opacity: torch.Tensor   # [1]