import torch
from dataclasses import dataclass

@dataclass
class CameraPose:
    position: torch.Tensor  # [3]
    quaternion: torch.Tensor  # [4] 
    intrinsics: torch.Tensor  # [3, 3]