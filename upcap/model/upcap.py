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
		self.register_buffer(
			'concepts_feat',
			torch.load(Cfg.concepts_feat_path, weights_only=True).float()
		)
	
	def forward(self, text_concepts: Tensor, token_ids: Tensor) -> Tensor:
		global_concept = text_concepts[:, :1]
		local_concepts = text_concepts[:, 1:]
		prefixes = self.attention(local_concepts, getattr(self, 'concepts_feat'))
		prefixes = torch.cat([global_concept, prefixes], dim=1)
		proj_prefixes = self.mlp(prefixes)
		text_embeds = self.gpt2.embed(token_ids)
		inputs = torch.cat([proj_prefixes, text_embeds], dim=1)
		return self.gpt2.forward_embeds(inputs)