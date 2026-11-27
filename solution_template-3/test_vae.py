import torch
from attacks.vae_attack import VAEAttack

attack = VAEAttack(n_avg_imgs=2, device="cpu")

img = torch.rand(3, 256, 256)
result = attack(img)

print("Shape:", result.shape)
print("Min:", result.min().item())
print("Max:", result.max().item())