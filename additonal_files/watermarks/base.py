import torch


class Watermark:
    """Base class for watermarks."""

    def __init__(self) -> None:
        pass

    def generate(self, prompt: str | list[str]) -> torch.Tensor:
        """Generate watermarked images images from prompts.

        TODO is there a better way to do this?

        Args:
            prompt: prompt/list of prompts for generation

        Returns:
            watermarked_img, (b,c,h,w) tensor, RGB channels in range [0,1]

        """

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Add watermark to an image.

        Args:
            img: input image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            watermarked_img, (b,c,h,w) tensor, RGB channels in range [0,1]

        """

    def decode(self, img: torch.Tensor) -> torch.Tensor:
        """Decode watermark from an image.

        Args:
            img: input image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            key: decoded keys, (b,*) tensor

        """

    def detect(self, img: torch.Tensor) -> torch.Tensor:
        """Detect if the image is watermarked or not.

        Args:
            img: input image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            pred: probability that the image is watermarked, (b) tensor, in range [0,1]

        """
