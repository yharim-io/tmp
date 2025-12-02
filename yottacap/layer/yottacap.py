import torch
from torch import nn, Tensor

from yottacap.layer.gpt2 import GPT2
from yottacap.layer.mlp import MLP
from yottacap.config import Cfg

class YottaCap(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.gpt2 = GPT2()
		self.mlp = MLP((Cfg.clip_dim, self.gpt2.emb_size))
	
	def get_emb_cat(self, clip_features: Tensor, token_ids: Tensor) -> Tensor:
		emb_img: Tensor = self.mlp(clip_features)
		emb_img = emb_img.reshape(-1, 1, self.gpt2.emb_size)
		emb_text: Tensor = self.gpt2.embed(token_ids)
		emb_cat: Tensor = torch.cat([emb_img, emb_text], dim=1)
		return emb_cat

	def forward_logits(self, clip_features: Tensor, token_ids: Tensor) -> Tensor:
		emb_cat: Tensor = self.get_emb_cat(clip_features, token_ids)
		return self.gpt2.forward_logits(emb_cat)

	def forward_hidden(self, clip_features: Tensor, token_ids: Tensor) -> Tensor:
		emb_cat: Tensor = self.get_emb_cat(clip_features, token_ids)
		return self.gpt2.forward_hidden(emb_cat)
	
	def forward(self, clip_features: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
		emb_cat: Tensor = self.get_emb_cat(clip_features, token_ids)
		return self.gpt2.forward(emb_cat)