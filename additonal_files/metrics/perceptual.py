import piq
import torch

from .base import Metric


class LPIPS(torch.nn.Module, Metric):
    def __init__(self, value_range: float = 1.) -> None:
        super().__init__()
        self.value_range = value_range
        self.lpips = piq.LPIPS(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.lpips(x / self.value_range, y / self.value_range)
