import torch
from torch import nn, Tensor

from .transformer_stack import TransformerStack

class TransformerMapper(nn.Module):
	
	def __init__(
		self,
		dim_clip: int,
		dim_emb: int,
		prefix_length: int,
		clip_length: int,
		num_layers: int = 8
	):
		self.clip_length = clip_length
		self.transformer = TransformerStack(dim_emb, 8, num_layers)
		self.linear = nn.Linear(dim_clip, clip_length * dim_emb)
		self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_emb), requires_grad=True)
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.linear(x).view(x.shape[0], self.clip_length, -1)
		prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
		prefix = torch.cat((x, prefix), dim=1)
		out = self.transformer(prefix)[:, self.clip_length:]
		return out