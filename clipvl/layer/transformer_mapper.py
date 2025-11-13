import torch
from torch import nn, Tensor

from .transformer_stack import TransformerStack

class TransformerMapper(nn.Module):
	
	def __init__(
		self,
		dim_vit: int,
		dim_emb: int,
		prefix_length: int,
		vit_seq_len: int,
		num_layers: int = 8
	):
		super().__init__()
		self.vit_seq_len = vit_seq_len
		self.transformer = TransformerStack(dim_emb, 8, num_layers)
		self.linear = nn.Linear(dim_vit, dim_emb)
		self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_emb), requires_grad=True)
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.linear(x) 
		prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape) # [B, 10, 768]
		prefix = torch.cat((x, prefix), dim=1) # [B, 50 + 10, 768]
		out = self.transformer(prefix)[:, self.vit_seq_len:] # [B, 10, 768]
		return out