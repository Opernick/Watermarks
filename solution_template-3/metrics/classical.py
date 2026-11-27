import piq
import torch
import torch.nn.functional as F
import vmaf_torch

from .base import Metric


class PSNR(torch.nn.Module, Metric):
    def __init__(self, value_range: float = 1.) -> None:
        super().__init__()
        self.value_range = value_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return piq.psnr(x, y, data_range=self.value_range, reduction="none")


class MSSSIM(torch.nn.Module, Metric):
    def __init__(self, value_range: float = 1.) -> None:
        super().__init__()
        self.value_range = value_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return piq.multi_scale_ssim(x, y, data_range=self.value_range, reduction="none")


class VMAF(torch.nn.Module, Metric):
    def __init__(self, value_range: float = 1., device="cpu") -> None:
        super().__init__()
        self.value_range = value_range
        self.vmaf = vmaf_torch.VMAF(temporal_pooling=False, enable_motion=False)

    @staticmethod
    def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
        r = image[..., 0:1, :, :]
        g = image[..., 1:2, :, :]
        b = image[..., 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y

    def forward(self, img_dist: torch.Tensor, img_ref: torch.Tensor) -> torch.Tensor:
        img_dist = self.rgb_to_y(img_dist) * (255. / self.value_range)
        img_ref = self.rgb_to_y(img_ref) * (255. / self.value_range)
        return self.vmaf(img_ref, img_dist).squeeze(-1)  # NOTE VMAF class expects inputs in order: ref, dist
