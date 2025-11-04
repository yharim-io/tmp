from torch import nn, Tensor
from torch.nn import functional as F

class TransformerFeedForward(nn.Module):
	
	def __init__(
		self,
		in_dim: int,
		h_dim: int,
		out_d: int | None,
		act = F.relu,
		dropout: float = 0.
	):
		super.__init__()
		out_d = out_d if out_d is not None else in_dim
		self.fc1 = nn.Linear(in_dim, h_dim)
		self.act = act
		self.fc2 = nn.Linear(h_dim, out_d)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x: Tensor) -> Tensor:
		x = self.fc1(x)
		x = self.act(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.dropout(x)
		return x