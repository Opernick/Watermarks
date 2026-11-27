# from wibench.algorithms.base import BaseAlgorithmWrapper
# from wibench.watermark_data import TorchBitWatermarkData
# from wibench.typing import TorchImg
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR += "/watermarks/wa/"
print(SCRIPT_DIR)
# sys.path.append(SCRIPT_DIR)

@dataclass
class WAParams:
    """Configuration parameters for the WA (Watermark Anything) watermarking algorithm.

    Attributes
    ----------
        wm_length : int
            Length of the watermark message to be embed (in bits).
        scaling_w : float
            Scaling factor for the watermark in the embedder model.

    """
    wm_length: int
    scaling_w: float


class WatermarkAnything():
    """Watermark Anything with Localized Messages - Image Watermarking Algorithm [`paper <https://arxiv.org/abs/2411.07231>`__].
    
    Provides an interface for embedding and extracting watermarks using the Watermark Anything watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/watermark-anything>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Watermark Anything algorithm configuration parameters

    """
    
    name = "watermark_anything"

    def __init__(
        self,
        # module_path: str,
        ckpt_path: str = "./wam_mit.pth",
        params_path: str = "./params.json",
        wm_length: int = 32,
        scaling_w: float = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # super().__init__(WAParams(wm_length=wm_length, scaling_w=scaling_w))
        # sys.path.append(SCRIPT_DIR)
        from .watermark_anything.data.metrics import msg_predict_inference

        from .notebooks.inference_utils import (
            load_model_from_checkpoint,
            normalize_img,
            unnormalize_img,
        )


        self.ckpt_path = ckpt_path
        self.params_path = params_path
        self.device = device
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} does not exist") 
        self.wam = load_model_from_checkpoint(
            self.params_path, self.ckpt_path
        ).to(device)
        self.wam.scaling_w = scaling_w
        self.transform = normalize_img
        self.msg_predict_inference = msg_predict_inference
        self.unnormalize_img = unnormalize_img


    def decode(self, image):
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (B, C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64

        """
        img = self.transform(image[0]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.wam.detect(img)["preds"].cpu()
        # [1, 256, 256], predicted mask
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits
        pred_message = self.msg_predict_inference(
            bit_preds, mask_preds
        ).float()
        return pred_message

    def embed(self, image, watermark_data):
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64

        """
        img = self.transform(image).unsqueeze(0).to(self.device)
        wm = watermark_data.to(self.device)
        with torch.no_grad():
            outputs = self.wam.embed(img, wm)
        result = outputs["imgs_w"].cpu()
        res = self.unnormalize_img(result.squeeze())
        return torch.clamp(res, 0, 1)
