import torch
from torch import nn, Tensor
from torch.nn import functional as F
import clip
from clip.model import CLIP

from .adapter import ImageAdapter, TextAdapter
from .discriminator import Discriminator
from .gpt2 import GPT2
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
	
	def gumbel_softmax(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
		# 1. Gumbel Noise (Reparameterization Trick)
		gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
		y = logits + gumbel_noise
		
		# 2. Softmax (Approximating discrete argmax)
		soft_one_hot = F.softmax(y / temperature, dim=-1) # (B, S, V)
		
		# 3. Soft Lookup in CLIP's Embedding Table
		vocab_size = self.clip_model.token_embedding.weight.shape[0]
		if soft_one_hot.shape[-1] > vocab_size:
			soft_one_hot = soft_one_hot[..., :vocab_size]
			
		soft_clip_embeds = soft_one_hot @ self.clip_model.token_embedding.weight.float()
		return soft_clip_embeds
	
	def softemb_to_clip(self, soft_embeds: Tensor) -> Tensor:
		# soft_embeds: (Batch, Seq, 512)
		batch_size, seq_len, dim = soft_embeds.shape
		target_len = self.clip_model.context_length  # 77
		
		# 1. Padding 到 77 (CLIP 标准长度)
		if seq_len < target_len:
			# 补零是安全的，因为 Causal Mask 会保证前面的 Token 看不到后面的 Padding
			padding = torch.zeros(batch_size, target_len - seq_len, dim, device=soft_embeds.device)
			x = torch.cat([soft_embeds, padding], dim=1) # (B, 77, D)
		else:
			x = soft_embeds[:, :target_len, :]
			
		# 2. 注入位置编码
		x = x + self.clip_model.positional_embedding.float()
		
		# 3. 维度调整 (B, S, D) -> (S, B, D)
		x = x.permute(1, 0, 2)
		
		# 4. 通过 CLIP Transformer (自动使用内置 Causal Mask)
		x = self.clip_model.transformer(x)
		
		# 5. 还原维度 (S, B, D) -> (B, S, D)
		x = x.permute(1, 0, 2)
		
		# 6. Layer Norm
		x = self.clip_model.ln_final(x)
		
		# 7. 截取回原始长度，丢弃 Padding 部分
		x = x[:, :seq_len, :]
		
		# 8. Mean Pooling (直接平均)
		# 你的要求：简单直接，利于收敛
		pooled = x.mean(dim=1) 
		
		# 9. 投影到联合空间
		features = pooled @ self.clip_model.text_projection.float()
		
		return features