import torch
from torch import nn, Tensor

from .mlp import MLP

class Discriminator(nn.Module):
	
	def __init__(self, input_dim: int):
		super().__init__()
		self.net = nn.Sequential(
			MLP((input_dim, input_dim // 2, 1)),
			nn.Sigmoid()
		)
	
	def forward(self, latents: Tensor) -> Tensor:
		pooled = latents.mean(dim=1)
		return self.net(pooled)