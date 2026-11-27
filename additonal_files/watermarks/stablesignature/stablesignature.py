import importlib
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from watermarks.base import Watermark


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    print(config.model)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class StableSignatureDecoder(Watermark):
    """Stable Signature watermark decoder based on the code from WAVES benchmark."""

    def __init__(self,
                 model_path:  str = "./models/StableSignature/stable_signature.onnx",
                 secret: np.ndarray | None = None,
                 device: str = "cpu",
                 device_id: int = 0,
                 ) -> None:
        super().__init__()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        self.device = device
        self.providers = ['CPUExecutionProvider'] if self.device == "cpu" else ['CUDAExecutionProvider']

        self.model = ort.InferenceSession(
            model_path,
            providers=self.providers,
            # providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(device_id)}],
            sess_options=session_options,
        )

        self.secret_len = 48
        if secret is None:
            # from WAVES.dev.constants.GROUND_TRUTH_MESSAGES["stable_sig"]
            self.secret = np.array([1., 1., 1., 0., 1., 0., 1., 1., 0., 1.,
                                    0., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
                                    0., 1., 1., 1., 0., 1., 0., 0., 1., 1.,
                                    0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
                                    0., 0., 1., 0., 0., 1., 1., 1.], dtype=np.float32)

        else:
            self.secret = np.array(secret, dtype=np.float32)
            assert self.secret.size == self.secret_len

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def decode(self, img: torch.Tensor) -> torch.Tensor:
        img = F.interpolate(img, size=(512, 512), mode="bicubic", antialias=True)  # [b,3,512,512]
        img = self.norm(img)
        img_np = img.detach().cpu().numpy()

        secret, = self.model.run(output_names=None, input_feed={"image": img_np})  # [b,secret_len]
        secret = torch.tensor(secret).to(img.device)  # [b,secret_len]
        secret = secret > 0  # [b,secret_len], bool
        return secret

    def detect(self, img: torch.Tensor) -> torch.Tensor:
        img_secret = self.decode(img) > 0  # [b,secret_len], bool
        true_secret = torch.tensor(self.secret).to(img.device).to(torch.bool)  # [secret_len]
        detect = (img_secret == true_secret).to(torch.float32).mean(dim=-1)  # [b]
        return detect
