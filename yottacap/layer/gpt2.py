from torch import nn, Tensor
from transformers import GPT2Config, GPT2LMHeadModel
import pickle

from yottacap.config import Cfg

class GPT2(nn.Module):
	"""
	4 layers transformer with 4 attention heads
	"""
	def __init__(self):
		super().__init__()
		with open(Cfg.gpt2_config_path, 'rb') as f:
			gpt2config: GPT2Config = pickle.load(f)
		self.core = GPT2LMHeadModel(gpt2config)
		self.emb_size = gpt2config.n_embd
		self.ember = self.core.get_input_embeddings()
	
	def forward_embeds(self, inputs_embeds: Tensor) -> Tensor:
		return self.core(inputs_embeds=inputs_embeds).logits
	
	def forward_hidden(self, inputs_embeds: Tensor) -> Tensor:
		return self.core.transformer(
			inputs_embeds = inputs_embeds
		).last_hidden_state
	
	def embed(self, token_ids: Tensor) -> Tensor:
		return self.ember(token_ids)
