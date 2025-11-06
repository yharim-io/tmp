from torch import nn, Tensor
from torch.nn import functional as F

from .attention import MultiHeadAttention
from .feed_forward import TransformerFeedForward

class TransformerBlock(nn.Module):
	
	def __init__(
		self,
		dim_self: int,
		dim_ref: int,
		num_heads: int,
		mlp_ratio: float = 4.,
		bias: bool = False,
		dropout: float = 0.,
		act = F.relu,
		norm_layer: nn.Module = nn.LayerNorm
	):
		super().__init__()
		self.norm1 = norm_layer(dim_self)
		self.attn = MultiHeadAttention(
			dim_self = dim_self,
			dim_ref = dim_ref,
			num_heads = num_heads,
			bias = bias,
			dropout = dropout
		)
		self.norm2 = norm_layer(dim_self)
		self.mlp = TransformerFeedForward(
			in_dim = dim_self,
			h_dim = int(dim_self * mlp_ratio),
			act = act,
			dropout = dropout
		)
	
	def forward(self, x: Tensor, y: Tensor|None=None, mask: Tensor|None=None):
		x = x + self.attn(self.norm1(x), y, mask)[0]
		x = x + self.mlp(self.norm2(x))
		return x
	
	def forward_with_attention(self, x: Tensor, y: Tensor|None=None, mask: Tensor|None=None):
		x_, attention = self.attn(self.norm1(x), y, mask)
		x += x_
		x += self.mlp(self.norm2(x))
		return x, attention
		
		