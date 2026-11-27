import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import torch.nn.functional as F

from attacks.base import Attack
from tqdm import tqdm

class VAEAttack(Attack):
    """
    VAE-based watermark removal attack.
    
    This attack uses a Variational Autoencoder (VAE) to encode the watermarked image
    into a latent space, adds noise to the latent representations, and then decodes
    back to image space. By averaging multiple noisy reconstructions, the attack
    aims to remove the watermark while preserving visual quality.
    """ 
    
    def __init__(
        self,
        n_avg_imgs: int = 2,
        noise_level: float = 0.5,
        device: str = "cuda:0",
        cache_dir: str | None = None,
        vae_name: str = "stabilityai/sd-vae-ft-mse"
    ) -> None:
        """
        Initialize VAE attack.
        
        Args:
            n_avg_imgs: Number of noisy reconstructions to average over
            noise_level: Standard deviation of noise to add to latent embeddings
            device: Device to run computations on (e.g., "cuda:0" or "cpu")
            cache_dir: Directory to cache pretrained models
            vae_name: Name of the pretrained VAE model to use
        """
        super().__init__()

        self.n_avg_imgs = n_avg_imgs
        self.noise_level = noise_level
        self.device = device
		# TODO Load the pretrained VAE model
        self.vae = AutoencoderKL.from_pretrained(
			vae_name,
			cache_dir = cache_dir
			).to(self.device)

        self.vae.eval()
        self.vae.requires_grad_(False)
        # TODO Preprocessing transform: normalize images to [-1, 1] range (expected by VAE), with params: [0.5], [0.5]
        self.normalize = transforms.Normalize(
            mean = [0.5],
            std = [0.5]
        )
    def add_noise_to_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add noise to latent embeddings.
        
        This method adds Gaussian noise to the latent representations to disrupt
        watermark patterns. The noise level is controlled by self.noise_level.
        
        Args:
            embeddings: Latent embeddings from VAE encoder, shape (batch, channels, height, width)
            
        Returns:
            Noisy embeddings with the same shape as input
            
        TODO: Implement this method to add Gaussian noise to the embeddings.
        - Scale the noise by self.noise_level

        """
        noise = torch.randn_like(embeddings)
        return embeddings + noise * self.noise_level
        
    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Perform VAE-based attack on input image(s)

        Args:
            img: Input image tensor, shape (batch, channels, height, width) or (channels, height, width)
                 Values should be in range [0, 1]
                 
        Returns:
            Attacked image tensor with same shape as input, values in range [0, 1]
            
        TODO: Implement the main attack logic
        """
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.to(self.device)

        normalized_img = self.normalize(img) # нормализация изображения

        embeddings = self.vae.encode(normalized_img).latent_dist.mode() # кодирование изображения

        reconstructed_imgs = []

        for _ in range(self.n_avg_imgs):
            attacked_embeddings = self.add_noise_to_embeddings(embeddings)
            attacked_img = self.vae.decode(attacked_embeddings).sample
            reconstructed_imgs.append(attacked_img)

        reconstructed_imgs = torch.stack(reconstructed_imgs) # преобразование в

        attacked_img = reconstructed_imgs.mean(dim=0)
        attacked_img = ((attacked_img + 1) / 2).clamp(0, 1)

        return attacked_img
