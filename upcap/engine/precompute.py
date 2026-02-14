import torch
import clip
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from clip.model import CLIP
from PIL import Image
from torchvision.transforms import Compose

from upcap.config import Cfg
from upcap.model.divider import Divider
from utils.tool import tqdm

@torch.inference_mode()
def compute_ref_text_feats_by_image_ids(
	image_ids: list[str],
	image_id_to_refs: dict[str, list[str]],
	clip_model: CLIP,
	batch_size: int = 4096,
) -> list[Tensor]:
	refs_by_image = [image_id_to_refs.get(image_id, []) for image_id in image_ids]
	ref_lens = [len(refs) for refs in refs_by_image]
	flat_refs = [ref for refs in refs_by_image for ref in refs]

	if not flat_refs:
		return [torch.empty((0, Cfg.clip_dim), dtype=torch.float16) for _ in ref_lens]

	all_ref_feats: list[Tensor] = []
	batch_starts = list(range(0, len(flat_refs), batch_size))
	iterator = tqdm(batch_starts, desc='Encoding Reference Text Features') if Cfg.is_master else batch_starts

	for start in iterator:
		batch_refs = flat_refs[start : start + batch_size]
		ref_tokens = clip.tokenize(batch_refs, truncate=True).to(Cfg.device)
		ref_feats = clip_model.encode_text(ref_tokens).float()
		ref_feats /= ref_feats.norm(dim=-1, keepdim=True)
		all_ref_feats.append(ref_feats.cpu().half())

	flat_ref_feats = torch.cat(all_ref_feats, dim=0)
	return list(torch.split(flat_ref_feats, ref_lens))

@torch.inference_mode()
def compute_concepts_local_feat(
	dataset: Dataset,
	divider: Divider,
	clip_model: CLIP,
	batch_size: int = 64,
	flatten: bool = True,
	use_distributed_sampler: bool = True,
) -> Tensor | list[Tensor]:
	
	sampler = DistributedSampler(dataset, shuffle=False) if use_distributed_sampler else None
	
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=sampler,
		num_workers=8,
		collate_fn=lambda x: x,
		pin_memory=True
	)
	
	all_features: list[Tensor] = []
	all_features_by_image: list[Tensor] = []
	
	mean = torch.tensor(Cfg.clip_mean, device=Cfg.device).view(1, 3, 1, 1).half()
	std = torch.tensor(Cfg.clip_std, device=Cfg.device).view(1, 3, 1, 1).half()
	
	if Cfg.is_master:
		iterator = tqdm(dataloader, desc='Computing Concept Features')
	else:
		iterator = dataloader
	
	for batch in iterator:
		image_paths = [item.get('image') for item in batch if item.get('image')]
		if not image_paths:
			continue

		if flatten:
			divided_images: Tensor = divider.process_batch(image_paths, flatten=True)
			
			if divided_images.numel() == 0:
				continue
			
			x = divided_images.permute(0, 3, 1, 2).half()
			x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
			x = x / 255.0
			x = (x - mean) / std
			
			features: Tensor = clip_model.encode_image(x)
			features /= features.norm(dim=-1, keepdim=True)
			
			all_features.append(features.cpu().half())
		else:
			divided_images_list = divider.process_batch(image_paths, flatten=False, output_size=224)
			for concept_images in divided_images_list:
				if concept_images.numel() == 0:
					all_features_by_image.append(torch.empty((0, Cfg.clip_dim), dtype=torch.float16))
					continue

				x = concept_images.permute(0, 3, 1, 2).half()
				x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
				x = x / 255.0
				x = (x - mean) / std

				features: Tensor = clip_model.encode_image(x)
				features /= features.norm(dim=-1, keepdim=True)
				all_features_by_image.append(features.cpu().half())

	if flatten:
		if not all_features:
			return torch.empty(0)
		return torch.cat(all_features, dim=0)

	return all_features_by_image

@torch.inference_mode()
def compute_concepts_global_feat(
	dataset: Dataset,
	clip_model: CLIP,
	preprocess: Compose,
	batch_size: int = 64,
	use_distributed_sampler: bool = True,
) -> Tensor:
	
	sampler = DistributedSampler(dataset, shuffle=False) if use_distributed_sampler else None
	
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