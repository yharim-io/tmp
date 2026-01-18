import torch
from torch import nn, Tensor
from enum import Enum

from .gpt2 import GPT2
from .mlp import MLP
from .transformer_mapper import TransformerMapper
from clipvl.config import Cfg

class MappingType(Enum):
	MLP = 'mlp'
	Transformer = 'transformer'

class ClipVLModel(nn.Module):

	def __init__(self, mapping_type: MappingType = MappingType.MLP):
		super().__init__()
		
		prefix_length = Cfg.prefix_length
		prefix_size = Cfg.vit_dim
		clip_length = Cfg.vit_seq_len
		num_layers = Cfg.num_layers
		
		self.gpt = GPT2()
		self.gpt_embedding_size = self.gpt.emb_size
		
		self.prefix_length = prefix_length
		# clip_length = clip_length if clip_length is not None else prefix_length
		
		if mapping_type == MappingType.MLP:
			mlp_prefix_size = prefix_size * clip_length
			self.clip_project = MLP(
				sizes = (
					mlp_prefix_size,
					(self.gpt_embedding_size * prefix_length) // 2,
					self.gpt_embedding_size * prefix_length
				)
			)
		else:
			self.clip_project = TransformerMapper(
				dim_vit = prefix_size,
				dim_emb = self.gpt_embedding_size,
				prefix_length = prefix_length,
				vit_seq_len = clip_length,
				num_layers = num_layers
			)
		
		self.mapping_type = mapping_type

	def get_dummy_token(self, batch_size: int, device: torch.device) -> Tensor:
		return torch.zeros(batch_size, Cfg.prefix_length, dtype=torch.int64, device=device)

	def forward(
		self,
		tokens: Tensor,
		prefix: Tensor, 
		mask: Tensor | None = None
	) -> Tensor:
		
		if self.mapping_type == MappingType.MLP:
			prefix = prefix.flatten(1)
			prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
		else:
			prefix_projections = self.clip_project(prefix)
		
		embedding_text = self.gpt.embed(tokens)
		
		embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
		
		if mask is not None:
			pass
		
		logits = self.gpt.forward_embeds(embedding_cat)
				
		return logits

class ClipVLPrefix(ClipVLModel):

	def parameters(self, recurse: bool = True):
		return self.clip_project.parameters()

	def train(self, mode: bool = True):
		super().train(mode)
		self.gpt.eval()
		return self
