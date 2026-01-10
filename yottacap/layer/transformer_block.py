from torch import nn, Tensor
from torch.nn import functional as F


from .attention import MultiHeadAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
	
	def __init__(
		self,
		dim: int,
		num_heads: int,
		mlp_ratio: float = 4.0,
		bias: bool = False,
		dropout: float = 0.0,
		act = nn.ReLU,
		norm_layer: nn.Module = nn.LayerNorm,
	):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = MultiHeadAttention(
			dim_self=dim,
			num_heads=num_heads,
			bias=bias,
			dropout=dropout,
		)
		self.norm2 = norm_layer(dim)
		self.mlp = MLP(
			sizes=(dim, int(dim * mlp_ratio), dim),
			bias=bias,
			act=act
		)
	
	def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
		x = x + self.attn(self.norm1(x), mask=mask)
		x = x + self.mlp(self.norm2(x))
		return x