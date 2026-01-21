import torch
from torch import nn, Tensor

from upcap.config import Cfg
from upcap.layer.mlp import MLP
from upcap.layer.gpt2 import GPT2
from upcap.layer.attention import ConceptAttention

class UpCap(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.global_attention = ConceptAttention(Cfg.clip_dim)
		self.local_attention = ConceptAttention(Cfg.clip_dim)
		self.gpt2 = GPT2()
		self.mlp = MLP((Cfg.clip_dim, self.gpt2.emb_size))
		
		# self.type_embedding = nn.Embedding(2, Cfg.clip_dim)
		# nn.init.normal_(self.type_embedding.weight, std=0.1) # [0.1, 0.5]

		concepts_global_feat_data = torch.load(Cfg.concepts_global_feat_path, weights_only=True).float()
		self.register_buffer('concepts_global_feat', concepts_global_feat_data, persistent=False)

		concepts_local_feat_data = torch.load(Cfg.concepts_local_feat_path, weights_only=True).float()
		self.register_buffer('concepts_local_feat', concepts_local_feat_data, persistent=False)

	def concepts_embed(
		self,
		text_concepts: Tensor,
		attn: bool = True
	) -> tuple[Tensor, Tensor]:
		
		# global_concept = text_concepts[:, :1]
		# local_concepts = text_concepts[:, 1:]

		# global_prefixes = self.global_attention(global_concept, self.concepts_global_feat)
		# local_prefixes = self.local_attention(local_concepts, self.concepts_local_feat)
		# prefixes = torch.cat([global_prefixes, local_prefixes], dim=1)

		# B, M, D = prefixes.shape
		# type_ids = torch.ones((B, M), dtype=torch.long, device=text_concepts.device)
		# type_ids[:, 0] = 0
		# prefixes += self.type_embedding(type_ids)

		# prefix_embeds = self.mlp(prefixes)

		# return prefix_embeds

		global_concept = text_concepts[:, :1]
		local_concepts = text_concepts[:, 1:]

		local_prefixes = self.local_attention(local_concepts, self.concepts_local_feat)

		if attn:
			global_prefixes = self.global_attention(global_concept, self.concepts_global_feat)
		else:
			global_prefixes = global_concept

		# B = text_concepts.shape[0]
		
		# global_type = torch.zeros((B, 1), dtype=torch.long, device=text_concepts.device)
		# global_prefixes += self.type_embedding(global_type)
		
		# local_type = torch.ones((B, local_prefixes.shape[1]), dtype=torch.long, device=text_concepts.device)
		# local_prefixes += self.type_embedding(local_type)

		global_embed = self.mlp(global_prefixes)
		local_embed = self.mlp(local_prefixes)

		return global_embed, local_embed

	def forward(self, text_concepts: Tensor, token_ids: Tensor) -> Tensor:
		# prefix_embeds = self.concepts_embed(text_concepts)
		# text_embeds = self.gpt2.embed(token_ids)
		# inputs = torch.cat([prefix_embeds, text_embeds], dim=1)
		# return self.gpt2.forward_embeds(inputs)

		# cross attention
		global_embed, local_embed = self.concepts_embed(text_concepts)
		text_embeds = self.gpt2.embed(token_ids)
		inputs = torch.cat([global_embed, text_embeds], dim=1)
		outputs = self.gpt2.forward_embeds(
			inputs_embeds=inputs,
			encoder_hidden_states=local_embed
		)
		return outputs