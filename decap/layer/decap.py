import torch
from torch import nn, Tensor

from decap.layer.gpt2 import GPT2
from decap.layer.mlp import MLP
from decap.config import Config

class DeCap(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.gpt2 = GPT2()
		self.mlp = MLP((Config.model.clip_dim, self.gpt2.emb_size))
	
	def forward(self, clip_features: Tensor, token_ids: Tensor):
		emb_img_prompt: Tensor = self.mlp(clip_features)
		emb_img_prompt = emb_img_prompt.reshape(-1, 1, self.gpt2.emb_size)
		emb_text: Tensor = self.gpt2.embed(token_ids)
		emb_cat: Tensor = torch.cat([emb_img_prompt, emb_text], dim=1)
		return self.gpt2.forward_embeds(emb_cat)