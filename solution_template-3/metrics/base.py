import torch

from datasets import ImageDataset


class Metric:
    """Base class for metrics."""

    def __init__(self) -> None:
        pass

    def __call__(self, img_dist: torch.Tensor, img_ref: torch.Tensor | None, *args, **kwargs) -> torch.Tensor:
        """Compute a quality metric for an image or batch of images.

        Args:
            img_dist: input distorted image, (b,c,h,w) tensor, RGB channels in range [0,1]
            img_ref: input reference image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            metric value, (b,) tensor

        """
        raise NotImplementedError("Subclasses of Metric should implement __call__.")

    def compute_on_dataloader(self,
                               dataloader_dist: torch.utils.data.DataLoader,
                               dataloader_ref: torch.utils.data.DataLoader | None,
                               reduction: str | None = "mean",
                               device: torch.device | str = "cuda",
                               *args,
                               **kwargs,
                               ) -> torch.Tensor:
        """Compute a quality metric for a dataloader of images.

        Args:
            dataloader_dist: dataloader with distorted images that yields (b,c,h,w) tensors, RGB channels in range [0,1]
            dataloader_ref: dataloader with reference images that yields (b,c,h,w) tensors, RGB channels in range [0,1]
            reduction: how to reduce computed metric values, one of {'none', 'mean', 'median'}
            device: device to compute with

        Returns:
            metric values, (N,) tensor where N is number of images in the dataloader or (1,) tensor with mean/median/etc. values

        """
        results = []
        if dataloader_ref is None: dataloader_ref = [None] * len(dataloader_dist)
        for img_dist, img_ref in zip(dataloader_dist, dataloader_ref, strict=True):
            img_dist = img_dist.to(device)
            if img_ref is not None: img_ref = img_ref.to(device)
            result = self(img_dist, img_ref, *args, **kwargs)
            results.append(result)
        results = torch.concat(results)
        return self.reduce(results, reduction)

    def compute_on_directory(self,
                               dir_dist: str,
                               dir_ref: str | None,
                               reduction: str | None = "mean",
                               device: torch.device | str = "cuda",
                               batch_size: int = 1,
                               *args,
                               **kwargs,
                               ) -> torch.Tensor:
        """Compute a quality metric for a directory of images.

        Args:
            dir_dist: dir with distorted images
            dir_ref: dir with reference images
            reduction: how to reduce computed metric values, one of {'none', 'mean', 'median'}
            device: device to compute with
            batch_size: batch size to use when computing metrics

        Returns:
            metric values, (N,) tensor where N is number of images in the directory or (1,) tensor with mean/median/etc. values

        """
        dataset_dist = ImageDataset(dir_dist, return_names=False)
        dataloader_dist = torch.utils.data.DataLoader(dataset_dist, batch_size=batch_size)

        if dir_ref:
            dataset_ref = ImageDataset(dir_ref, return_names=False)  #TODO remove dependency on ImageDataset?
            dataloader_ref = torch.utils.data.DataLoader(dataset_ref, batch_size=batch_size)
        else:
            dataloader_ref = None

        return self.compute_on_dataloaders(dataloader_dist, dataloader_ref, reduction, device, *args, **kwargs)

    @staticmethod
    def reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        if reduction == "none" or reduction is None:
            return x
        if reduction == "mean":
            return x.mean(dim=0)
        if reduction == "median":
            return x.median(dim=0)
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'median'}")
