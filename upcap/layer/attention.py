import torch
from torch import nn, Tensor

class ConceptAttention(nn.Module):
	
	def __init__(self, dim: int):
		super().__init__()
		self.q_proj = nn.Linear(dim, dim, bias=False)
		self.k_proj = nn.Linear(dim, dim, bias=False)
		self.v_proj = nn.Linear(dim, dim, bias=False)
		self.scale = dim ** -0.5
		
	def forward(self, text_concepts: Tensor, concept_bank: Tensor) -> Tensor:
		Q = self.q_proj(text_concepts)
		K = self.k_proj(concept_bank)
		V = self.v_proj(concept_bank)
		
		attn_score = (Q @ K.transpose(-2, -1)) * self.scale
		attn_probs = attn_score.softmax(dim=-1)
		
		return attn_probs @ V