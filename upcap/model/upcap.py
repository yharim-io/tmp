import torch
from torch import nn, Tensor

from upcap.config import Cfg
from upcap.layer.mlp import MLP
from upcap.layer.gpt2 import GPT2
from upcap.layer.attention import ConceptAttention

class UpCap(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.attention = ConceptAttention(Cfg.clip_dim)
		self.gpt2 = GPT2()
		self.mlp = MLP((Cfg.clip_dim, self.gpt2.emb_size))
		self.register_buffer('concept_bank', torch.load(Cfg.concepts_feat_path, weights_only=True))
	
	def forward(self, text_concepts: Tensor, token_ids: Tensor) -> Tensor:
		prefixes = self.attention(text_concepts, self.concept_bank)
		proj_prefixes = self.mlp(prefixes)
		text_embeds = self.gpt2.embed(token_ids)
		inputs = torch.cat([proj_prefixes, text_embeds], dim=1)
		return self.gpt2.forward_embeds(inputs)