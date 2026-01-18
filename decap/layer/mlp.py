from torch import Tensor
from torch import nn

class MLP(nn.Module):
	
	def __init__(self, sizes: tuple[int, ...], bias=True, act=nn.Tanh):
		super().__init__()
		layers = []
		num_layers = len(sizes)
		for i in range(1, num_layers):
			layers.append(nn.Linear(sizes[i - 1], sizes[i], bias=bias))
			if i != num_layers - 1:
				layers.append(act())
		self.model = nn.Sequential(*layers)
	
	def forward(self, x: Tensor) -> Tensor:
		return self.model(x)