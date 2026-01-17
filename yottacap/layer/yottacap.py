import torch
from torch import nn, Tensor
from torch.nn import functional as F
import clip

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
		self.discriminator = Discriminator(Cfg.latent_dim, Cfg.latent_seq_len)
		self.gpt2 = GPT2()
		
		self.feature_hook_data = {}
		self._register_hooks()
	
	def _register_hooks(self):
		def image_hook(module, input, output):
			self.feature_hook_data['vit_tokens'] = output.permute(1, 0, 2)
		def text_hook(module, input, output):
			self.feature_hook_data['text_tokens'] = output.permute(1, 0, 2)
		self.clip_model.visual.transformer.register_forward_hook(image_hook)
		self.clip_model.transformer.register_forward_hook(text_hook)
	
	@torch.no_grad()
	def extract_clip_features(self, image = None, text = None) -> dict:
		result = {}
		self.feature_hook_data.clear()
		
		if image is not None:
			image_feat = self.clip_model.encode_image(image)
			result['T_image'] = image_feat / image_feat.norm(dim=-1, keepdim=True)
			result['vit_tokens'] = self.feature_hook_data['vit_tokens'].float()
		if text is not None:
			text_feat = self.clip_model.encode_text(text)
			result['T_text'] = text_feat / text_feat.norm(dim=-1, keepdim=True)
			result['text_tokens'] = self.feature_hook_data['text_tokens'].float()
		return result

	def forward(self, latents: Tensor, token_ids: Tensor) -> Tensor:
		text_emb = self.gpt2.embed(token_ids)
		emb_cat = torch.cat([latents, text_emb], dim=1)
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
	
	def gumbel_softmax_st(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
		y_hard = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
		vocab_size = self.clip_model.token_embedding.weight.shape[0]
		if y_hard.shape[-1] > vocab_size:
			y_hard = y_hard[..., :vocab_size]
		soft_clip_embeds = y_hard @ self.clip_model.token_embedding.weight.float()
		return soft_clip_embeds

	def softemb_to_clip(self, soft_embeds: Tensor) -> Tensor:
		# soft_embeds: (Batch, Seq, 512)
		batch_size, seq_len, dim = soft_embeds.shape
		target_len = self.clip_model.context_length
		
		# 1. 准备 SOT & EOT Embedding
		sot_token = torch.tensor([Cfg.sot_token_id], device=soft_embeds.device)
		sot_emb = self.clip_model.token_embedding(sot_token).float()
		sot_emb = sot_emb.unsqueeze(0).expand(batch_size, -1, -1) # (B, 1, D)

		eot_token = torch.tensor([Cfg.eos_token_id], device=soft_embeds.device)
		eot_emb = self.clip_model.token_embedding(eot_token).float()
		eot_emb = eot_emb.unsqueeze(0).expand(batch_size, -1, -1) # (B, 1, D)
		
		# 2. 拼接: [SOT, Soft_Tokens, EOT]
		x = torch.cat([sot_emb, soft_embeds, eot_emb], dim=1)
		curr_len = x.shape[1]
		eot_idx = seq_len + 1
		
		# 3. 长度处理 (Padding 或 Truncate)
		if curr_len < target_len:
			padding = torch.zeros(batch_size, target_len - curr_len, dim, device=soft_embeds.device)
			x = torch.cat([x, padding], dim=1)
		elif curr_len > target_len:
			x = x[:, :target_len, :]
			# 强制将最后一位覆盖为 EOT，确保聚合点存在
			x[:, target_len - 1, :] = eot_emb.squeeze(1)
			eot_idx = target_len - 1
			
		# 4. CLIP Forward
		x = x + self.clip_model.positional_embedding.float()
		x = x.permute(1, 0, 2)
		x = self.clip_model.transformer(x)
		x = x.permute(1, 0, 2)
		x = self.clip_model.ln_final(x)
		
		# 5. 提取 EOT 特征并投影
		features = x[:, eot_idx, :] @ self.clip_model.text_projection.float()
		return features