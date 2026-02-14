import torch
from torch import nn, Tensor

from upcap.config import Cfg
from upcap.model.noise import VelocityField
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

		if self.training:
			self.noise = VelocityField.load_from_pretrained(
				checkpoint_path=Cfg.noise_rf_path,
				map_location='cpu'
			)
			self.noise.requires_grad_(False)
			self.noise.eval()
		
		self.prefix_len: int = 0

	def _transport_features(self, feat: Tensor) -> Tensor:
		shape = feat.shape
		flat_feat = feat.reshape(-1, shape[-1])
		flat_feat = self.noise.transport(flat_feat)
		return flat_feat.view(*shape)

	def embed_tokens(self, token_ids: Tensor) -> Tensor:
		return self.gpt2.embed(token_ids)

	def project_features(
		self,
		global_feat: Tensor,
		local_feat: Tensor,
		global_attn: bool = False,
		local_attn: bool = False
	) -> tuple[Tensor, Tensor]:

		# if self.training:
		# 	global_feat = self._transport_features(global_feat)
		if global_attn:
			global_feat = self.global_attention(global_feat, self.concepts_global_feat)
		global_emb = self.mlp(global_feat)

		# if self.training:
		# 	local_feat = self._transport_features(local_feat)
		if local_attn:
			local_feat = self.local_attention(local_feat, self.concepts_local_feat)
		local_emb = self.mlp(local_feat)
		
		return global_emb, local_emb

	def assemble_structure(
		self,
		global_emb: Tensor,
		local_emb: Tensor,
		text_emb: Tensor,
	) -> tuple[Tensor, Tensor | None]:
		
		inputs_embeds = torch.cat([global_emb, text_emb], dim=1)
		encoder_hidden_states = torch.cat([local_emb], dim=1)

		self.prefix_len = inputs_embeds.shape[1] - text_emb.shape[1]

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