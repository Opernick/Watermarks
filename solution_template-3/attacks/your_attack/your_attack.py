import torch
from attacks.base import Attack

class YourAttack(Attack):
   

    def __init__(self) -> None:
        
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError("You need to implement the forward method")