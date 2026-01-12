import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
	
	def __init__(self, latent_dim: int, latent_seq_len: int):
		super().__init__()
		flat_dim = latent_dim * latent_seq_len
		self.net = nn.Sequential(
			# Layer 1: Flatten Dim (7680) -> Input Dim (768)
			# 将展开的序列特征压缩回潜在维度，提取全局模式
			spectral_norm(nn.Linear(flat_dim, latent_dim)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			
			# Layer 2: Input Dim (768) -> Hidden (384)
			# 进一步抽象特征
			spectral_norm(nn.Linear(latent_dim, latent_dim // 2)),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.1),
			
			# Layer 3 (Output): Hidden (384) -> 1
			# 输出 Logits (不含 Sigmoid，配合 BCEWithLogitsLoss)
			spectral_norm(nn.Linear(latent_dim // 2, 1))
		)
	
	def forward(self, latents: Tensor) -> Tensor:
		# latents: [Batch, Seq_Len, Dim]
		flat_latents = latents.flatten(start_dim=1)# [Batch, Seq_Len * Dim]
		return self.net(flat_latents)