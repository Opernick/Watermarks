import torch


# TODO make this an abstract base class?
class Attack(torch.nn.Module):
    """Base class for attacks."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Attack an image or batch of images.

        Args:
            img: input image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            attacked_img, (b,c,h,w) tensor, RGB channels in range [0,1]

        """
        pass