from torch import nn, Tensor
from transformers import GPT2LMHeadModel

from zerocap.config import Cfg

class GPT2(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.core = GPT2LMHeadModel.from_pretrained(Cfg.gpt2_pretrained_path)
		gpt2config = self.core.config
		self.emb_size = gpt2config.n_embd
		self.ember = self.core.get_input_embeddings()
	
	def forward_embeds(self, inputs_embeds: Tensor) -> Tensor:
		return self.core(inputs_embeds=inputs_embeds).logits
	
	def embed(self, token_ids: Tensor) -> Tensor:
		return self.ember(token_ids)