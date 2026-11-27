import torch
import torchvision
from diffusers.models import AutoencoderKL
from transformers import AutoProcessor, CLIPModel


class ClipEmbedding(torch.nn.Module):
    """
    CLIP embedding extractor.
    
    This class extracts image embeddings using the CLIP (Contrastive Language-Image Pre-training)
    model. CLIP embeddings are learned representations that encode semantic information about images.
    These embeddings are useful for watermark removal attacks because they capture high-level
    visual features that may be more robust to watermarks than pixel-level features.
    """
    
    def __init__(self, device):
        """
        Initialize CLIP embedding model.
        
        Args:
            device: Device to run the model on (CPU or CUDA)
        """
        super().__init__()
        self.device = device
        
        # TODO Load pretrained CLIP model and processor (use clip-vit-base-patch32 model)
        self.model = ...
        self.processor = ...
        
        # CLIP-specific normalization parameters
        # These are the mean and std values used during CLIP training
        openai_clip_mean = [0.48145466, 0.4578275, 0.40821073]
        openai_clip_std = [0.26862954, 0.26130258, 0.27577711]
        
        # TODO Create preprocessing transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP embeddings from input images.
        
        Args:
            x: Input images, shape (batch, channels, height, width), values in [0, 1]
            
        Returns:
            Image embeddings, shape (batch, embedding_dim)
            
        TODO: Implement the forward pass
        """
        
        raise NotImplementedError("You need to implement the forward method")


class ResNet18Embedding(torch.nn.Module):
    """
    ResNet18 embedding extractor.
    
    This class extracts image embeddings using a pretrained ResNet18 model.
    
    The model supports extracting features from layers:
    - "last": Features before the final classification layer (most commonly used)

    """
    
    def __init__(self, layer: str, device: str):
        """
        Initialize ResNet18 embedding model.
        
        Args:
            layer: Which layer to extract features
            device: Device to run the model on (CPU or CUDA)
        """
        super().__init__()
        self.device = device

        # TODO Load pretrained ResNet18 model
        original_model = ...
        
        # ImageNet normalization parameters
        # These are the mean and std values used during ImageNet training
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        if layer == "last":
            # TODO Extract everything except the final fully connected layer
            # This gives us features after global average pooling
            self.features = ...
        else:
            raise ValueError("Invalid layer name")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract ResNet18 embeddings from input images.
        
        Args:
            images: Input images, shape (batch, channels, height, width), values in [0, 1]
            
        Returns:
            Image embeddings, shape depends on the layer:
            - For "last": (batch, 512) - flattened features after global average pooling
            - For other layers: (batch, channels, height, width) - spatial feature maps
            
        TODO: Implement the forward pass
        """
        # Hint: Resize images to 224x224
        
        raise NotImplementedError("You need to implement the forward method")
