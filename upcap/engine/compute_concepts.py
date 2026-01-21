import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import os
os.environ["TQDM_NCOLS"] = "40"
from tqdm import tqdm
from clip.model import CLIP
from PIL import Image
from torchvision.transforms import Compose

from upcap.config import Cfg
from upcap.model.divider import Divider

@torch.inference_mode()
def compute_concepts_local_image(
	dataset: Dataset,
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
	
	all_images: list[Tensor] = []
	
	mean = torch.tensor(Cfg.clip_mean, device=Cfg.device).view(1, 3, 1, 1).half()
	std = torch.tensor(Cfg.clip_std, device=Cfg.device).view(1, 3, 1, 1).half()
	
	if Cfg.is_master:
		iterator = tqdm(dataloader, desc='Extracting Concept Images')
	else:
		iterator = dataloader
	
	for batch in iterator:
		image_paths = [item.get('image') for item in batch if item.get('image')]
		if not image_paths:
			continue

		divided_images: Tensor = divider.process_batch(image_paths, flatten=True)
		
		if divided_images.numel() == 0:
			continue
		
		x = divided_images.permute(0, 3, 1, 2).half()
		x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
		x = x / 255.0
		x = (x - mean) / std
		
		all_images.append(x.cpu())

	if not all_images:
		return torch.empty(0)

	return torch.cat(all_images, dim=0)

@torch.inference_mode()
def compute_concepts_local_feat(
	clip_model: CLIP,
	batch_size: int = 512
) -> Tensor:
	
	images = torch.load(Cfg.concepts_local_image_path, map_location='cpu', weights_only=True)
	dataset = TensorDataset(images)
	
	sampler = DistributedSampler(dataset, shuffle=False)
	
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=sampler,
		num_workers=8,
		pin_memory=True
	)
	
	num_samples = len(sampler)
	feat_dim = Cfg.clip_dim
	all_features = torch.empty((num_samples, feat_dim), dtype=torch.float16, device='cpu', pin_memory=True)
	
	if Cfg.is_master:
		iterator = tqdm(dataloader, desc='Computing Concept Features')
	else:
		iterator = dataloader
	
	start_idx = 0
	
	for batch in iterator:
		img_batch = batch[0].to(Cfg.device, non_blocking=True)
		
		features: Tensor = clip_model.encode_image(img_batch)
		features /= features.norm(dim=-1, keepdim=True)
		
		batch_len = features.shape[0]
		end_idx = start_idx + batch_len
		all_features[start_idx : end_idx] = features.cpu().half()
		start_idx = end_idx

	return all_features

@torch.inference_mode()
def compute_concepts_global_feat(
	dataset: Dataset,
	clip_model: CLIP,
	preprocess: Compose,
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
	
	if Cfg.is_master:
		iterator = tqdm(dataloader, desc='Computing Global Features')
	else:
		iterator = dataloader
	
	for batch in iterator:
		image_paths = [item.get('image') for item in batch if item.get('image')]
		if not image_paths:
			continue
			
		images = [preprocess(Image.open(p)) for p in image_paths]
		if not images:
			continue
			
		img_tensor = torch.stack(images).to(Cfg.device)
		
		features = clip_model.encode_image(img_tensor)
		features /= features.norm(dim=-1, keepdim=True)
		
		all_features.append(features.cpu().half())
		
	if not all_features:
		return torch.empty(0)
		
	return torch.cat(all_features, dim=0)