import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import numpy as np
from tqdm import tqdm
from clip.model import CLIP
from torchvision.transforms import Compose

from upcap.config import Cfg
from upcap.model.divider import Divider

@torch.no_grad()
def compute_concepts(
	dataset: Dataset,
	clip_model: CLIP,
	preprocess: Compose,
	divider: Divider,
	batch_size: int = 64
) -> Tensor:
	
	sampler = DistributedSampler(dataset, shuffle=False)
	
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=sampler,
		num_workers=8,
		collate_fn=lambda x: x, 
		pin_memory=True
	)
	
	all_features: list[Tensor] = []
	
	# CLIP 归一化常数 (GPU)
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=Cfg.device).view(1, 3, 1, 1)
	std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=Cfg.device).view(1, 3, 1, 1)
	
	if Cfg.is_master:
		iterator = tqdm(dataloader, desc='Computing Concepts')
	else:
		iterator = dataloader
	
	for batch in iterator:
		image_paths = [item.get('image') for item in batch if item.get('image')]
		if not image_paths:
			continue

		# (N, H, W, 3) Float Tensor on GPU, range 0-255
		divided_images: Tensor = divider.process_batch(image_paths)
		
		if divided_images.numel() == 0:
			continue
		
		# 优化：直接在 GPU 上并行处理，替代 CPU 循环
		# Permute to (N, 3, H, W)
		x = divided_images.permute(0, 3, 1, 2)
		# Resize to CLIP input size (224)
		x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
		# Normalize
		x = x / 255.0
		x = (x - mean) / std
		
		num_concepts = x.shape[0]
		
		for i in range(0, num_concepts, batch_size):
			clip_batch = x[i : i + batch_size]
			
			features: Tensor = clip_model.encode_image(clip_batch)
			features /= features.norm(dim=-1, keepdim=True)
			
			all_features.append(features.cpu())

	if not all_features:
		return torch.empty(0)

	return torch.cat(all_features, dim=0)