import torch
from torch import Tensor
from torch import nn

class MultiHeadAttention(nn.Module):
	
	def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim_self // num_heads
		self.scale = head_dim ** -0.5
		self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
		self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
		self.project = nn.Linear(dim_self, dim_self)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x: Tensor, y: Tensor|None=None, mask: Tensor|None=None) -> tuple[Tensor, ...]:
		y = y if y is not None else x
		b, n, c = x.shape
		_, m, d = y.shape
		# b n h dh
		queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
		# b m 2 h dh
		keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
		keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
		attention: Tensor = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
		
		if mask is not None:
			if mask.dim() == 2:
				mask.unsqueeze_(1)
			attention = attention.masked_fill(mask.unsqueeze(3), float('-inf'))
		
		attention = attention.softmax(dim=2)
		out: Tensor = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
		out = self.project(out)
		return out, attention