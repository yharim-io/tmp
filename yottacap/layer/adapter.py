import torch
from torch import nn, Tensor

from yottacap.config import Cfg
from .transformer_stack import TransformerStack

class AdapterBase(nn.Module):
	def __init__(self):
		super().__init__()
		self.transformer = TransformerStack(
			dim = Cfg.latent_dim,
			num_heads = Cfg.adapter_heads,
			num_layers = Cfg.adapter_depth,
		)

class ImageAdapter(AdapterBase):
	def forward(self, features: Tensor) -> Tensor:
		return self.transformer(features)

class TextAdapter(AdapterBase):
	def __init__(self):
		super().__init__()
		self.project = nn.Linear(Cfg.clip_dim, Cfg.latent_dim)
		self.mask_token = nn.Parameter(torch.randn(1, 1, Cfg.latent_dim))
	
	def random_mask(self, x: Tensor) -> Tensor:
		B, L, D = x.shape
		mask = torch.rand(B, L, device=x.device) < Cfg.text_mask_ratio
		mask_expanded = mask.unsqueeze(-1).expand_as(x)
		masked_x = torch.where(mask_expanded, self.mask_token.expand_as(x), x)
		return masked_x
	
	def forward(self, features: Tensor) -> Tensor:
		x = self.project(features)
		masked_x = self.random_mask(x)
		return self.transformer(masked_x)