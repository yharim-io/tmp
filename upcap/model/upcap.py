import torch
from torch import nn, Tensor

from upcap.config import Cfg
from upcap.layer.mlp import MLP
from upcap.layer.gpt2 import GPT2
from upcap.layer.attention import ConceptAttention

class UpCap(nn.Module):
	
	def __init__(
		self,
		enable_concepts_global_buffer: bool = True,
		enable_concepts_local_buffer: bool = True,
	):
		super().__init__()
		self.global_attention = ConceptAttention(Cfg.clip_dim)
		self.local_attention = ConceptAttention(Cfg.clip_dim)
		self.gpt2 = GPT2()
		self.mlp = MLP((Cfg.clip_dim, self.gpt2.emb_size))
		
		# self.type_embedding = nn.Embedding(2, Cfg.clip_dim)
		# nn.init.normal_(self.type_embedding.weight, std=0.1) # [0.1, 0.5]

		if enable_concepts_global_buffer:
			concepts_global_feat_data = torch.load(Cfg.concepts_global_feat_path, weights_only=True).float()
			self.register_buffer('concepts_global_feat', concepts_global_feat_data, persistent=False)
			

		if enable_concepts_local_buffer:
			concepts_local_feat_data = torch.load(Cfg.concepts_local_feat_path, weights_only=True).float()
			self.register_buffer('concepts_local_feat', concepts_local_feat_data, persistent=False)

	def concepts_embed(
		self,
		text_concepts: Tensor,
		global_attn: bool = False,
		local_attn: bool = False,
		cross_attn: bool = False,
	) -> tuple[Tensor, Tensor]:
		
		# def noise(x: Tensor) -> Tensor:
		# 	x = x.norm(dim=-1, keepdim=True)
		# 	_f = (1 - x.pow(2)).clamp(min=1.14514e-6).sqrt()
		# 	_g = torch.rand_like(x)
		# 	_g = _g / _g.norm(dim=-1, keepdim=True)
		# 	return _f * _g
		
		global_concept = text_concepts[:, :1]
		local_concepts = text_concepts[:, 1:]

		if global_attn:
			global_concept = self.global_attention(global_concept, self.concepts_global_feat)
			# global_prefixes = global_prefixes + noise(global_prefixes)
		global_embed = self.mlp(global_concept)

		if not cross_attn:
			return global_embed, torch.empty(0)

		if local_attn:
			local_concepts = self.local_attention(local_concepts, self.concepts_local_feat)
		local_embed = self.mlp(local_concepts)
		
		return global_embed, local_embed

	def forward(
		self,
		text_concepts: Tensor,
		token_ids: Tensor,
		global_attn: bool = False,
		local_attn: bool = False,
		cross_attn: bool = False
	) -> Tensor:

		global_embed, local_embed = self.concepts_embed(
			text_concepts,
			global_attn=global_attn,
			local_attn=local_attn,
			cross_attn=cross_attn
		)
		text_embeds = self.gpt2.embed(token_ids)
		inputs = torch.cat([global_embed, text_embeds], dim=1)
		if cross_attn:
			outputs = self.gpt2.forward_embeds(
				inputs_embeds=inputs,
				encoder_hidden_states=local_embed
			)
		else:
			outputs = self.gpt2.forward_embeds(inputs_embeds=inputs)
		return outputs