import torch
from dataclasses import dataclass

@dataclass
class GaussianPrimitive:
    position: torch.Tensor  # [3]
    rotation: torch.Tensor  # [4] quaternion
    scale: torch.Tensor     # [3]
    color: torch.Tensor     # [3]
    opacity: torch.Tensor   # [1]