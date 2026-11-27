import torch
import numpy as np
import sys, os

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

# from wibench.watermark_data import TorchBitWatermarkData
# from wibench.algorithms.base import BaseAlgorithmWrapper
# from wibench.typing import TorchImg
# from wibench.utils import (
#     resize_torch_img,
#     overlay_difference,
#     normalize_image,
#     denormalize_image
# )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# SCRIPT_DIR += "/watermarks/arwgan/"
# print(SCRIPT_DIR)

def resize_torch_img(image, size, mode: str = 'bilinear', align_corners: bool = True):
    """Resize a torch image tensor to specified dimensions.
    
    Parameters
    ----------
    image : TorchImg
        Input image tensor in (C, H, W) format
    size : List[int]
        Target size as [height, width]
    mode : str, optional
        Interpolation mode ('nearest', 'bilinear', 'bicubic')
        Default is 'bilinear'
    align_corners : bool, optional
        Flag for align_corners parameter in interpolation
        Default is True
        
    Returns
    -------
    TorchImg
        Resized image tensor
    """
    if list(image.shape)[1:] == size:
        return image.clone()
    if mode in ['bilinear', 'bicubic']:
        image = image.unsqueeze(0)
    resized_image = torch.nn.functional.interpolate(image, size, mode=mode, align_corners=align_corners).squeeze(0)
    return resized_image


def overlay_difference(original_image, resized_image, marked_image):
    """Overlay difference between images of one size to image of another size.
    
    Parameters
    ----------
    original_image : TorchImg
        Base reference image to overlay on
    resized_image : TorchImg
        Resized version of original (should match marked_image size)
    marked_image : TorchImg
        Watermarked or processed image
        
    Returns
    -------
    TorchImg
        Original image with overlay
        
    Notes
    -----
    - Computes difference between marked and resized images
    - Resizes difference to match original image dimensions
    - Adds difference to original image
    """
    orig_height, orig_width = original_image.shape[1:]
    diff = marked_image - resized_image
    min_val = diff.min()
    diff_resized = resize_torch_img((diff - min_val).squeeze(0), (orig_height, orig_width))
    marked_image = torch.clip(original_image + diff_resized + min_val, 0, 1).squeeze(0)
    return marked_image


def normalize_image(image, transform = None):
    """Normalize image tensor from [0,1] to [-1,1] range and add batch dimension.
    
    Parameters
    ----------
    image : TorchImg
        Input image in [0,1] range (C, H, W)
    transform : Optional[Normalize], optional
        Custom normalization transform. If None, uses default [-1,1] scaling.
        
    Returns
    -------
    TorchImgNormalize
        Normalized image in [-1,1] range with batch dimension (1, C, H, W)
    """
    if transform is not None:
        return transform(image).unsqueeze(0)
    return (image * 2 - 1).unsqueeze(0)


def denormalize_image(image, transform = None):
    """Denormalize image tensor from [-1,1] to [0,1] range and remove batch dimension.
    
    Parameters
    ----------
    image : TorchImgNormalize
        Input image in [-1,1] range (1, C, H, W)
    transform : Optional[Normalize], optional
        Custom denormalization transform. If None, uses default scaling.
        
    Returns
    -------
    TorchImg
        Denormalized image in [0,1] range (C, H, W)
    """
    if transform is not None:
        return transform(image).squeeze(0)
    return ((image + 1) / 2).squeeze(0)

@dataclass
class ARWGANParams:
    """
    Configuration parameters for the `ARWGAN <https://ieeexplore.ieee.org/document/10155247>`__ watermarking algorithm.

    Attributes
    ----------
        H : int
            Height of the input image (in pixels). Determines the vertical size of image tensors
        W : int
            Width of the input image (in pixels). Determines the horizontal size of image tensors
        wm_length : int
            Length of the binary watermark message to embed (in bits)

        encoder_blocks : int
            Number of convolutional blocks in the encoder network
        encoder_channels : int
            Number of filters (channels) in each encoder block

        decoder_blocks : int
            Number of convolutional blocks in the decoder network
        decoder_channels : int
            Number of filters in each decoder block

        use_discriminator : bool
            If True, enables the use of an adversarial discriminator
        use_vgg : bool
            If True, adds a perceptual loss using VGG features to improve

        discriminator_blocks : int
            Number of convolutional blocks in the discriminator network
        discriminator_channels : int
            Number of filters in each discriminator block

        decoder_loss : float
            Weight of the decoder loss term in the total loss function. Controls the importance of accurate message recovery
        encoder_loss : float
            Weight of the encoder loss term in the total loss function. Typically regularizes visual similarity between original and encoded images
        adversarial_loss : float
            Weight of the adversarial loss term in the total loss. Higher values push the encoder to generate more realistic images when a discriminator is used

        enable_fp16 : bool
            If True, enables mixed precision (fp16) training/inference for improved speed and reduced memory usage on compatible hardware (default False)

    """
    H: int
    W: int
    wm_length: int
    encoder_blocks: int
    encoder_channels: int
    decoder_blocks: int
    decoder_channels: int
    use_discriminator: bool
    use_vgg: bool
    discriminator_blocks: int
    discriminator_channels: int
    decoder_loss: float
    encoder_loss: float
    adversarial_loss: float
    enable_fp16: bool = False


class ARWGAN():
    """
    `ARWGAN <https://ieeexplore.ieee.org/document/10155247>`__: Attention-Guided Robust Image Watermarking Model Based on GAN --- Image Watermarking Algorithm.
    
    Provides an interface for embedding and extracting watermarks using the ARWGAN watermarking algorithm.
    Based on the code from `here <https://github.com/river-huang/ARWGAN>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        ARWGAN algorithm configuration parameters

    """

    name = "arwgan"

    # /watermarks/arwgan/
    
    def __init__(
        self, 
        ckpt_path: str = "./models/ARWGAN/ARWGAN.pyt",
        params_path: str = "./models/ARWGAN/options-and-config.pickle",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:

        # sys.path.append(params["module_path"])
        # sys.path.append(SCRIPT_DIR)
        # os.chdir(SCRIPT_DIR)
        from utils import load_options
        from model.encoder_decoder import EncoderDecoder
        from noise_layers.noiser import Noiser

        options_file_path = params_path
        checkpoint_file_path = ckpt_path

        if options_file_path is None:
            raise FileNotFoundError(f"The options file path: '{options_file_path}' does not exist!")
        if checkpoint_file_path is None:
            raise FileNotFoundError(f"The yaml config path: '{checkpoint_file_path}' does not exist!")

        options_file_path = Path(options_file_path).resolve()
        checkpoint_file_path = Path(checkpoint_file_path).resolve()
        train_options, config, noise_config = load_options(options_file_path)
        
        self.device = device
        checkpoint = torch.load(checkpoint_file_path, map_location=self.device)

        self.params = ARWGANParams(
            H=config.H,
            W=config.W,
            wm_length=config.message_length,
            encoder_blocks=config.encoder_blocks,
            encoder_channels=config.encoder_channels,
            decoder_blocks=config.decoder_blocks,
            decoder_channels=config.decoder_channels,
            discriminator_blocks=config.discriminator_blocks,
            discriminator_channels=config.discriminator_channels,
            decoder_loss=config.decoder_loss,
            encoder_loss=config.encoder_loss,
            adversarial_loss=config.adversarial_loss,
            enable_fp16=config.enable_fp16,
            use_discriminator=config.use_discriminator,
            use_vgg = config.use_vgg
        )
        # super().__init__(params)

        noiser = Noiser([], self.device)
        self.encoder_decoder = EncoderDecoder(config, noiser)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder = self.encoder_decoder.to(self.device)
        self.encoder_decoder.eval()

    def embed(self, image, watermark_data):
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalized_image = normalize_image(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(resized_normalized_image.to(self.device), watermark_data.watermark.to(self.device))
        denormalized_marked_image = denormalize_image(encoded_tensor.cpu())
        marked_image = overlay_difference(image, resized_image, denormalized_marked_image)
        return marked_image
    
    def decode(self, image):
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (B, C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized_image = resize_torch_img(image[0], (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(resized_normalize_image.to(self.device))
        return (res.cpu().numpy() > 0.5).astype(int)
    
    # def watermark_data_gen(self) -> TorchBitWatermarkData:
    #     """Generate watermark payload data for ARWGAN watermarking algorithm.
        
    #     Returns
    #     -------
    #     TorchBitWatermarkData
    #         Torch bit message with data type torch.int64 and shape of (0, message_length)

    #     Notes
    #     -----
    #     - Called automatically during embedding
    #     """
    #     return TorchBitWatermarkData.get_random(self.params.wm_length)
