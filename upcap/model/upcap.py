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

		if enable_concepts_global_buffer:
			concepts_global_feat_data = torch.load(Cfg.concepts_global_feat_path, weights_only=True).float()
			self.register_buffer('concepts_global_feat', concepts_global_feat_data, persistent=False)

		if enable_concepts_local_buffer:
			concepts_local_feat_data = torch.load(Cfg.concepts_local_feat_path, weights_only=True).float()
			self.register_buffer('concepts_local_feat', concepts_local_feat_data, persistent=False)

	def embed_tokens(self, token_ids: Tensor) -> Tensor:
		return self.gpt2.embed(token_ids)

	def project_features(
		self,
		global_feat: Tensor,
		local_feat: Tensor,
		global_attn: bool = False,
		local_attn: bool = False
	) -> tuple[Tensor, Tensor]:
		
		def noise(x: Tensor) -> Tensor:
			norm = x.norm(dim=-1, keepdim=True)
			delta = (1 - norm.pow(2)).clamp(min=1e-6).sqrt()
			rand = torch.randn_like(x)
			proj = (rand * x).sum(dim=-1, keepdim=True) / norm.pow(2).clamp(min=1e-6) * x
			ortho = rand - proj
			ortho = ortho / ortho.norm(dim=-1, keepdim=True).clamp(min=1e-6)
			return delta * ortho

		if global_attn:
			global_feat = self.global_attention(global_feat, self.concepts_global_feat)
			global_feat = global_feat + noise(global_feat)
		global_emb = self.mlp(global_feat)

		if local_feat.numel() > 0:
			if local_attn:
				local_feat = self.local_attention(local_feat, self.concepts_local_feat)
				local_feat = local_feat + noise(local_feat)
			local_emb = self.mlp(local_feat)
		else:
			local_emb = torch.empty(0, device=global_feat.device)
		
		return global_emb, local_emb

	def assemble_structure(
		self,
		global_emb: Tensor,
		local_emb: Tensor,
		text_emb: Tensor,
	) -> tuple[Tensor, Tensor | None]:
		# inputs_embeds = torch.cat([global_emb, text_emb], dim=1)
		# encoder_hidden_states = local_emb
		inputs_embeds = text_emb
		encoder_hidden_states = global_emb
		return inputs_embeds, encoder_hidden_states

	def forward(
		self,
		inputs_embeds: Tensor,
		encoder_hidden_states: Tensor | None = None,
		past_key_values: tuple | None = None
	) -> tuple[Tensor, tuple]:
		return self.gpt2.forward_embeds(
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			past_key_values=past_key_values,
			use_cache=True
		)