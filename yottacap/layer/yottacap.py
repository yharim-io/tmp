import torch
from torch import nn, Tensor
from torch.nn import functional as F
import clip
from clip.model import CLIP

from .adapter import ImageAdapter, TextAdapter
from .discriminator import Discriminator
from .gpt2 import GPT2
from .mlp import MLP
from yottacap.config import Cfg

class YottaCap(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.clip_model, _ = clip.load(Cfg.clip_pretrained_path, device='cpu', jit=False)
		self.clip_model.eval()
		for p in self.clip_model.parameters():
			p.requires_grad = False
		
		self.image_adapter = ImageAdapter()
		self.text_adapter = TextAdapter()
		self.discriminator = Discriminator(Cfg.latent_dim)
		self.gpt2 = GPT2()
		
		if Cfg.latent_dim != self.gpt2.emb_size:
			self.latent_proj = nn.Linear(Cfg.latent_dim, self.gpt2.emb_size)
		else:
			self.latent_proj = nn.Identity()
		
		self.feature_hook_data = {}
		self._register_hooks()
	
	def _register_hooks(self):
		def image_hook(module, input, output):
			self.feature_hook_data['vit_tokens'] = output.permute(1, 0, 2)
		def text_hook(module, input, output):
			self.feature_hook_data['text_tokens'] = output.permute(1, 0, 2)
		self.clip_model.visual.transformer.register_forward_hook(image_hook)
		self.clip_model.transformer.register_forward_hook(text_hook)
	
	def extract_clip_features(self, image = None, text = None) -> dict:
		result = {}
		self.feature_hook_data.clear()
		
		if image is not None:
			with torch.no_grad():
				image_feat = self.clip_model.encode_image(image)
				result['T_image'] = image_feat / image_feat.norm(dim=-1, keepdim=True)
				result['vit_tokens'] = self.feature_hook_data['vit_tokens'].float()
		if text is not None:
			with torch.no_grad():
				text_feat = self.clip_model.encode_text(text)
				result['T_text'] = text_feat / text_feat.norm(dim=-1, keepdim=True)
				result['text_tokens'] = self.feature_hook_data['text_tokens'].float()
		return result
	
	def get_image_latent(self, vit_tokens: Tensor) -> Tensor:
		return self.image_adapter(vit_tokens)

	def get_text_latent(self, text_tokens: Tensor) -> Tensor:
		return self.text_adapter(text_tokens)

	def forward(self, latents: Tensor, token_ids: Tensor) -> Tensor:
		prefix = self.latent_proj(latents)
		text_emb = self.gpt2.embed(token_ids)
		emb_cat = torch.cat([prefix, text_emb], dim=1)
		return self.gpt2.forward_logits(emb_cat)
	
	def gumbel_softmax(self, logits: Tensor, temperature: float) -> Tensor:
		# 1. Gumbel Noise
		gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
		y = logits + gumbel_noise
		
		# 2. Softmax
		soft_one_hot = F.softmax(y / temperature, dim=-1) # (B, S, V)
		
		# 3. Multiply with Embedding Matrix
		# GPT2 embedding weight: (V, D)
		# (B, S, V) @ (V, D) -> (B, S, D)
		soft_embeds = soft_one_hot @ self.gpt2.core.wte.weight
		return soft_embeds
	
	def project_to_clip(self, soft_embeds: Tensor) -> Tensor:
		pooled = soft_embeds.mean(dim=1)
		return pooled @ self.clip_model.text_projection