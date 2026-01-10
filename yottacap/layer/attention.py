import torch
from torch import Tensor
from torch import nn

class MultiHeadAttention(nn.Module):
	
	def __init__(
		self,
		dim_self,
		num_heads,
		bias=True,
		dropout=0.0,
	):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim_self // num_heads
		self.scale = head_dim ** -0.5
		self.to_q = nn.Linear(dim_self, dim_self, bias=bias)
		self.to_k = nn.Linear(dim_self, dim_self, bias=bias)
		self.to_v = nn.Linear(dim_self, dim_self, bias=bias)
		self.project = nn.Linear(dim_self, dim_self)
		self.dropout = nn.Dropout(dropout)
	
	def forward(
		self,
		x: Tensor,
		mask: Tensor | None = None,
	) -> Tensor:
		B, N, C = x.shape
		
		q = self.to_q(x).reshape(B, N, self.num_heads, C // self.num_heads)
		k = self.to_k(x).reshape(B, N, self.num_heads, C // self.num_heads)
		v = self.to_v(x).reshape(B, N, self.num_heads, C // self.num_heads)

		attention = torch.einsum('bnhd,bmhd->bhnm', q, k) * self.scale
		
		if mask is not None:
			attention = attention.masked_fill(mask == 0, float('-inf'))
		
		attention = attention.softmax(dim=-1)
		
		out: Tensor = torch.einsum('bhnm,bmhd->bnhd', attention, v).reshape(B, N, C)
		out = self.project(out)
		out = self.dropout(out)
		return out