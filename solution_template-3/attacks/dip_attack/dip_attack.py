import numpy as np
import torch
from tqdm import tqdm

from attacks.base import Attack

from .model_dip import get_net_dip


def get_model(dig_cfgs):
    """
    Get a DIP (Deep Image Prior) model.
    
    Args:
        dig_cfgs: Dictionary containing model configuration, must include "arch" key
        
    Returns:
        Initialized DIP model in training mode
    """
    if dig_cfgs["arch"] == "vanila":
        dip_model = get_net_dip(dig_cfgs["arch"])
    else:
        raise RuntimeError("Unsupported DIP architecture.")
    dip_model.train()
    return dip_model


class DIPAttack(Attack):
    """
    DIP-based watermark evasion attack.
    
    Deep Image Prior (DIP) is a technique that uses the structure of a convolutional
    neural network as a prior for natural images. This attack optimizes a randomly
    initialized network to reconstruct the input image, which can remove watermarks
    while preserving the main content.
    
    The key insight is that the network architecture itself acts as a regularization
    that favors natural images over watermarks, which are often more structured.
    
    NOTE: This version uses the original image as input (non-randomized), which is
    slightly different from the original DIP paper but is commonly used in practice.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32,
        total_iters: int = 100,
        lr: float = 0.01,
        arch: str = "vanila",
    ) -> None:
        """
        Initialize DIP attack.

        Args:
            device: Device to run computations on (CPU or CUDA)
            dtype: Data type for computations (float32 or float16)
            total_iters: Total number of optimization iterations
            lr: Learning rate for the optimizer
            arch: DIP architecture type (currently only "vanila" is supported)
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.total_iters = total_iters
        self.lr = lr
        self.arch = arch

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform DIP-based attack on input image(s).
        
        
        Args:
            img: Input image tensor, shape (batch, channels, height, width) or (channels, height, width)
                 Values should be in range [0, 1]
                 
        Returns:
            Attacked image tensor with same shape as input, values in range [0, 1]
            
        TODO: Implement the DIP attack optimization loop
        """
        
        raise NotImplementedError("You need to implement the forward method")


def fill_noise(x, noise_type):
    """
    Fill tensor `x` with noise of type `noise_type`.
    
    This is a helper function for generating noise inputs for DIP models.
    
    Args:
        x: Tensor to fill with noise
        noise_type: Type of noise - "u" for uniform, "n" for normal
    """
    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        assert False


def np_to_torch(img_np):
    """
    Convert image in numpy.array to torch.Tensor.
    
    Converts from C x W x H [0..1] to C x W x H [0..1] with batch dimension.
    
    Args:
        img_np: Numpy array of shape (C, H, W)
        
    Returns:
        Torch tensor of shape (1, C, H, W)
    """
    return torch.from_numpy(img_np)[None, :]


def get_noise(input_depth, method, spatial_size, noise_type="u", var=1.0 / 10):
    """
    Generate noise tensor for DIP model input.
    
    Return a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    
    Args:
        input_depth: Number of channels in the tensor
        method: "noise" for filling tensor with noise; "meshgrid" for np.meshgrid
        spatial_size: Spatial size of the tensor to initialize (height, width) or single int
        noise_type: 'u' for uniform; 'n' for normal (only used if method="noise")
        var: A factor by which noise will be multiplied (standard deviation scaler)
        
    Returns:
        Initialized tensor of shape (1, input_depth, spatial_size[0], spatial_size[1])
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == "noise":
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1), 
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1)
        )
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


class DIPAttackNoise(Attack):
    """
    DIP-based watermark evasion attack with correct noise input.
    
    This is a more correct implementation that follows the original DIP paper,
    where the network input is random noise rather than the original image.
    This version uses noise initialization as described in the original DIP implementation.
    
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype: torch.dtype = torch.float32,
        total_iters: int = 200,
        lr: float = 0.003,
        arch: str = "vanila",
        input_noise_method: str = "n",
        input_noise_var: float = 1.0 / 10,
    ) -> None:
        """
        Initialize DIP attack with noise input.

        Args:
            device: Device to run computations on (CPU or CUDA)
            dtype: Data type for computations (float32 or float16)
            total_iters: Total number of optimization iterations
            lr: Learning rate for the optimizer
            arch: DIP architecture type (currently only "vanila" is supported)
            input_noise_method: Method for generating noise input ("n" for normal, "u" for uniform)
            input_noise_var: Variance/scaling factor for the noise input
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.total_iters = total_iters
        self.lr = lr
        self.arch = arch
        self.input_noise_method = input_noise_method
        self.input_noise_var = input_noise_var

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform DIP-based attack with noise input on input image(s).
        
        The attack process is similar to DIPAttack, but uses random noise as the
        network input instead of the original image. This follows the original
        DIP paper more closely.
        
        Args:
            img: Input image tensor, shape (batch, channels, height, width) or (channels, height, width)
                 Values should be in range [0, 1]
                 
        Returns:
            Attacked image tensor with same shape as input, values in range [0, 1]
            
        TODO: Implement the DIP attack with noise input
        """
        
        raise NotImplementedError("You need to implement the forward method")
