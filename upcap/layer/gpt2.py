from torch import nn, Tensor
from transformers import GPT2Config, GPT2LMHeadModel
import pickle

from upcap.config import Cfg

class GPT2(nn.Module):
	def __init__(self):
		super().__init__()
		with open(Cfg.gpt2_config_path, 'rb') as f:
			gpt2config: GPT2Config = pickle.load(f)
		gpt2config.add_cross_attention = True
		gpt2config._attn_implementation = 'eager'
		self.core = GPT2LMHeadModel(gpt2config)
		self.emb_size = gpt2config.n_embd
		self.ember = self.core.get_input_embeddings()
	
	def forward_embeds(
		self,
		inputs_embeds: Tensor,
		encoder_hidden_states = None,
		past_key_values = None,
		use_cache: bool = False
	):
		out = self.core(
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			past_key_values=past_key_values,
			use_cache=use_cache
		)
		if use_cache:
			return out.logits, out.past_key_values
		else:
			return out.logits
	
	def embed(self, token_ids: Tensor) -> Tensor:
		return self.ember(token_ids)