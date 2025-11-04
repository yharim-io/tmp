from torch import nn, Tensor
from torch.nn import functional as F

from .transformer_block import TransformerBlock

class TransformerStack(nn.Module):
	
	def __init__(
		self,
		dim_self: int,
		num_heads: int,
		num_layers: int,
		dim_ref: int | None = None,
		mlp_ratio: float = 2.,
		act = F.relu,
		norm_layer: nn.Module = nn.LayerNorm,
		enc_dec: bool = False
	):
		super().__init__()
		dim_ref = dim_ref if dim_ref is not None else dim_self
		self.enc_dec = enc_dec
		if enc_dec:
			num_layers = num_layers * 2
		layers = []
		for i in range(num_layers):
			if i % 2 == 0 and enc_dec:
				layers.append(TransformerBlock(
					dim_self = dim_self,
					dim_ref = dim_ref,
					num_heads = num_heads,
					mlp_ratio = mlp_ratio,
					act = act,
					norm_layer = norm_layer
				))
			elif enc_dec:
				layers.append(TransformerBlock(
					dim_self = dim_self,
					dim_ref = dim_self,
					num_heads = num_heads,
					mlp_ratio = mlp_ratio,
					act = act,
					norm_layer = norm_layer
				))
			else:
				layers.append(TransformerBlock(
					dim_self = dim_self,
					dim_ref = dim_ref,
					num_heads = num_heads,
					mlp_ratio = mlp_ratio,
					act = act,
					norm_layer = norm_layer
				))
		self.layers = nn.ModuleList(layers)
	
	def forward(self, x: Tensor, y: Tensor|None=None, mask: Tensor|None=None) -> Tensor:
		for i, layer in enumerate(self.layers):
			if i % 2 == 0 and self.enc_dec:	# cross
				x = layer(x, y)
			elif self.enc_dec:	# self
				x = layer(x, x, mask)
			else:	# self or cross
				x = layer(x, y, mask)
		return x
	
	def forward_with_attention(self, x: Tensor, y: Tensor|None=None, mask: Tensor|None=None) -> Tensor:
		attentions = []
		for layer in self.layers:
			x, att = layer.forward_with_attention(x, y, mask)
			attentions.append(att)
		return x, attentions
