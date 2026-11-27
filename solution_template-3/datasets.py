"""Dataset classes."""

import os

import torch
import torchvision
from natsort import natsorted
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    """Dataset created using a single directory with images.

    Args:
        img_dir: path to image directory
        return_names: if True will return tuple (image_name, image_tensor), if False will return image tensor. Defaults to True.
        value_range: pixel values will be in range [0, value_range]. Defaults to 1..
        subset: if integer only first n images in the directory will be used, if None all images in the directory will be used. Defaults to None.

    """

    def __init__(self,
                 img_dir: str,
                 return_names: bool = True,
                 value_range: float = 1.,
                 subset: int | None = None,
                 ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.return_names = return_names
        self.value_range = value_range
        # sort in natural order 1.png, 2.png, 3.png, ... instead of lexicographical 1.png, 10.png, 100.png, ...
        self.img_names = natsorted(os.listdir(img_dir))
        if subset:
            self.img_names = self.img_names[:subset]

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[str, torch.Tensor]:
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = torchvision.transforms.functional.pil_to_tensor(Image.open(img_path))
        img = img.to(torch.float32) * (self.value_range / 255.)
        return (img_name, img) if self.return_names else img

