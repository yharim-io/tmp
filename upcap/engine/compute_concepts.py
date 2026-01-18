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
	batch_size: int = 128
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
	
	if Cfg.is_master:
		iterator = tqdm(dataloader, desc='Computing Concepts')
	else:
		iterator = dataloader
	
	for batch in iterator:
		image_paths = []
		for item in batch:
			p = item.get('image')
			if p:
				image_paths.append(p)
		
		if not image_paths:
			continue

		divided_images: Tensor = divider.process_batch(image_paths)
		
		if divided_images.numel() == 0:
			continue
		
		imgs_np = divided_images.cpu().numpy().astype(np.uint8)
		clip_inputs = []
		
		for img_arr in imgs_np:
			img_pil = Image.fromarray(img_arr)
			processed_img = preprocess(img_pil)
			clip_inputs.append(processed_img)
		
		if not clip_inputs:
			continue

		total_clip_inputs = torch.stack(clip_inputs)
		num_concepts = total_clip_inputs.shape[0]
		
		for i in range(0, num_concepts, batch_size):
			clip_batch = total_clip_inputs[i : i + batch_size].to(Cfg.device)
			
			features: Tensor = clip_model.encode_image(clip_batch)
			features /= features.norm(dim=-1, keepdim=True)
			
			all_features.append(features.cpu())

	if not all_features:
		return torch.empty(0)

	return torch.cat(all_features, dim=0)