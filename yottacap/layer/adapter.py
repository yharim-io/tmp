import torch
from torch import nn, Tensor

from yottacap.config import Cfg
from .transformer_stack import TransformerStack

class AdapterBase(nn.Module):
	def __init__(self):
		super().__init__()
		self.target_len = Cfg.latent_seq_len
		
		self.transformer = TransformerStack(
			dim=Cfg.latent_dim,
			num_heads=Cfg.adapter_heads,
			num_layers=Cfg.adapter_depth,
		)
		
		self.latent_queries = nn.Parameter(torch.randn(1, self.target_len, Cfg.latent_dim))
		nn.init.normal_(self.latent_queries, std=0.02)

	def forward_reduce(self, x: Tensor) -> Tensor:
		B = x.shape[0]
		queries = self.latent_queries.expand(B, -1, -1)
		combined_seq = torch.cat([x, queries], dim=1)
		out = self.transformer(combined_seq)
		return out[:, -self.target_len:, :]

class ImageAdapter(AdapterBase):
	def forward(self, features: Tensor) -> Tensor:
		return self.forward_reduce(features)

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
		return self.forward_reduce(masked_x)