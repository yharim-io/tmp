import torch
from torch import nn, Tensor

class ConceptAttention(nn.Module):
	
	def __init__(self, dim: int):
		super().__init__()
		# self.q_proj = nn.Linear(dim, dim, bias=False)
		# self.k_proj = nn.Linear(dim, dim, bias=False)
		# self.v_proj = nn.Linear(dim, dim, bias=False)
		self.scale = dim ** -0.5
		
	def forward(self, text_concepts: Tensor, concepts_feat: Tensor) -> Tensor:
		# Q: Tensor = self.q_proj(text_concepts)
		# K: Tensor = self.k_proj(concepts_feat)
		# V: Tensor = self.v_proj(concepts_feat)
		
		# attn_score: Tensor = (Q @ K.transpose(-2, -1)) * self.scale
		# attn_probs: Tensor = attn_score.softmax(dim=-1)
		
		# o: Tensor = attn_probs @ V
		
		# return o
		
		attn_score: Tensor = (text_concepts @ concepts_feat.transpose(-2, -1)) * self.scale
		attn_probs: Tensor = attn_score.softmax(dim=-1)
		o = attn_probs @ concepts_feat
		
		return o