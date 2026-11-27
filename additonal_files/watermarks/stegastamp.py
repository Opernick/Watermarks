import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from watermarks.base import Watermark


class Stegastamp(Watermark):
    """Stegastamp watermark based on code from WAVES benchmark.

    NOTE PIL resize was changed to torch interpolate
    TODO is resizing here correct? crop would be better?
    TODO should receive secret as input in encode?
    """

    def __init__(self,
                 stegastamp_model_path: str = "./models/StegaStamp/stega_stamp.onnx",
                 secret: np.ndarray | None = None,
                 device: str = "cpu",
                 ) -> None:
        super().__init__()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.log_severity_level = 3
        self.device = device
        self.providers = ['CPUExecutionProvider'] if self.device == "cpu" else ['CUDAExecutionProvider']
        self.stegastamp_model = ort.InferenceSession(
            stegastamp_model_path,
            providers=self.providers,
            provider_options=[{"device_id": "0"}],
            sess_options=session_options,
        )

        self.secret_len = 100
        if secret is None:
            # from WAVES.dev.constants.GROUND_TRUTH_MESSAGES["stegastamp"]
            self.secret = np.array([1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 1.,
                                    1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 0.,
                                    0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1.,
                                    0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
                                    1., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 1., 0., 1.],
                                   dtype=np.float32)
        else:
            self.secret = np.array(secret, dtype=np.float32)
            assert self.secret.size == self.secret_len

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        img = F.interpolate(img, size=(400, 400), mode="bicubic", antialias=True)  # [b,c,400,400]
        img_np = img.permute(0, 2, 3, 1).detach().cpu().numpy()  # [b,400,400,3]

        secret = np.broadcast_to(self.secret, (b, self.secret_len))
        attacked_img, attacked_residual, dummy_secret = self.stegastamp_model.run(output_names=None, input_feed={"image": img_np, "secret": secret})

        attacked_img = torch.tensor(attacked_img).permute(0, 3, 1, 2).to(img.device)
        attacked_img = F.interpolate(attacked_img, size=(h, w), mode="bicubic")
        return attacked_img

    def decode(self, img: torch.Tensor) -> torch.Tensor:
        img = F.interpolate(img, size=(400, 400), mode="bicubic", antialias=True)  # [b,c,400,400]
        img_np = img.permute(0, 2, 3, 1).detach().cpu().numpy()  # [b,400,400,3]

        dummy_secret = np.zeros((img.shape[0], self.secret_len), dtype=np.float32)  # not used by the model, onnx needs this to work
        dummy_img, dummy_residual, secret = self.stegastamp_model.run(output_names=None, input_feed={"image": img_np, "secret": dummy_secret})
        secret = torch.tensor(secret).to(img.device)  # [b,secret_len]
        return secret

    def detect(self, img: torch.Tensor) -> torch.Tensor:
        img_secret = self.decode(img)  # [b,secret_len]
        true_secret = torch.tensor(self.secret).to(img.device)  # [secret_len]
        pred = (img_secret == true_secret).to(torch.float32).mean(dim=-1)  # [b]
        return pred

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.encode(img)
