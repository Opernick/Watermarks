import numpy as np
import torch
from tqdm import tqdm

from attacks.base import Attack

from .feature_extractors import ClipEmbedding, ResNet18Embedding


class AdversarialEmbedding(Attack):
    """
    Adversarial embedding attack from WAVES benchmark.
    
    This attack modifies images to change their embeddings in a feature space
    (e.g., CLIP or ResNet embeddings) while keeping the visual appearance similar.
    The goal is to remove watermarks by making the attacked image have a different
    embedding than the original, which can disrupt watermark detection that relies
    on these embeddings.
    
    The attack uses Projected Gradient Descent (PGD) to iteratively modify the image
    to minimize the distance between the original and attacked embeddings, subject
    to L-infinity norm constraints on the pixel changes.
    """

    def __init__(self,
                 device: torch.device | str,
                 encoder: str,
                 loss_type: str = "l2",  # metric between attacked and non-attacked embeddings
                 strength: int = 9,  # 2,4,6,8
                 eps_factor: float = 1 / 255,
                 alpha_factor: float = 0.1,
                 n_steps: int = 200,
                 random_start: bool = True,
                 ) -> None:
        """
        Initialize adversarial embedding attack.
        
        Args:
            device: Device to run computations on (CPU or CUDA)
            encoder: Type of embedding model to use ("resnet18" or "clip")
            loss_type: Loss function type ("l1" or "l2") for comparing embeddings
            strength: Attack strength multiplier
            eps_factor: Base epsilon factor for perturbation bound
            alpha_factor: Step size factor relative to epsilon
            n_steps: Number of PGD optimization steps
            random_start: Whether to start from a random perturbation
        """
        super().__init__()

        # Load embedding model
        self.encoder = encoder

        if encoder == "resnet18":
            # We use the last layer's state as the embedding
            embedding_model = ResNet18Embedding("last", device)
        elif encoder == "clip":
            embedding_model = ClipEmbedding(device)
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        embedding_model = embedding_model.to(device)
        embedding_model.eval()

        self.model = embedding_model
        self.device = device
        self.eps = eps_factor * strength  # Maximum perturbation bound (L-infinity)
        self.alpha = alpha_factor * eps_factor * strength  # Step size for PGD
        self.steps = n_steps
        self.loss_type = loss_type
        self.random_start = random_start

        # TODO Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = ...
        elif self.loss_type == "l2":
            self.loss_fn = ...
        else:
            raise ValueError("Unsupported loss type")

    def pgd(self, images: torch.Tensor, init_delta: torch.Tensor = None) -> torch.Tensor:
        """
        Perform Projected Gradient Descent (PGD) attack.
        
        Args:
            images: Input images, shape (batch, channels, height, width), values in [0, 1]
            init_delta: Optional initial perturbation (if not using random start)
            
        Returns:
            Adversarial images with same shape as input, values in [0, 1]
            
        TODO: Implement the PGD attack
        """
        
        raise NotImplementedError("You need to implement the pgd method")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attack.
        
        Args:
            img: Input image tensor, shape (batch, channels, height, width) or (channels, height, width)
                 Values should be in range [0, 1]
                 
        Returns:
            Attacked image tensor with same shape as input, values in range [0, 1]
        """
        if len(img.shape) < 4:
            img = img.unsqueeze(0)

        return self.pgd(img)
