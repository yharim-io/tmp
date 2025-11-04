import torch
from torch import nn, Tensor
from enum import Enum

from .gpt2 import GPT2
from .mlp import MLP
from .transformer_mapper import TransformerMapper
from clipcap.config import Config

class MappingType(Enum):
	MLP = 'mlp'
	Transformer = 'transformer'

class ClipCaptionModel(nn.Module):

	def __init__(
		self,
		prefix_length: int,
		clip_length: int | None = None,
		prefix_size: int = 512,
		num_layers: int = 8,
		mapping_type: MappingType = MappingType.MLP
	):
		super().__init__()
		
		self.gpt = GPT2()
		self.gpt_embedding_size = self.gpt.emb_size
		
		self.prefix_length = prefix_length
		clip_length = clip_length if clip_length is not None else prefix_length
		
		if mapping_type == MappingType.MLP:
			self.clip_project = MLP(
				sizes = (
					prefix_size,
					(self.gpt_embedding_size * prefix_length) // 2,
					self.gpt_embedding_size * prefix_length
				)
			)
		else:
			self.clip_project = TransformerMapper(
				dim_clip = prefix_size,
				dim_emb = self.gpt_embedding_size,
				prefix_length = prefix_length,
				clip_length = clip_length,
				num_layers = num_layers
			)

	def get_dummy_token(self, batch_size: int, device: torch.device) -> Tensor:
		return torch.zeros(batch_size, Config.model.prefix_length, dtype=torch.int64, device=device)

	def forward(
		self,
		tokens: Tensor,
		prefix: Tensor, 
		mask: Tensor | None = None,
		labels: Tensor | None = None
	):
		
		prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
		
		embedding_text = self.gpt.embed(tokens)
		
		embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
		
		if mask is not None:
			pass
		
		logits = self.gpt.forward_embeds(embedding_cat)
		
		class ModelOutput:
			def __init__(self, logits_tensor):
				self.logits = logits_tensor
				
		return ModelOutput(logits=logits)


class ClipCaptionPrefix(ClipCaptionModel):

	def parameters(self, recurse: bool = True):
		return self.clip_project.parameters()

	def train(self, mode: bool = True):
		super().train(mode)
		self.gpt.eval()
		return self
