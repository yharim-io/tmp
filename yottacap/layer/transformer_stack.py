from torch import nn, Tensor

from .transformer_block import TransformerBlock

class TransformerStack(nn.Module):
	
	def __init__(
		self,
		dim: int,
		num_heads: int,
		num_layers: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
	):
		super().__init__()
		self.layers = nn.ModuleList([
			TransformerBlock(
				dim=dim,
				num_heads=num_heads,
				mlp_ratio=mlp_ratio,
				dropout=dropout,
			)
			for _ in range(num_layers)
		])
	
	def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
		for layer in self.layers:
			x = layer(x, mask=mask)
		return x