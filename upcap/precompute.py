import torch
import torch.distributed as dist
import clip
import gc
from torch.utils.data import Dataset
from pathlib import Path

from upcap.config import Cfg
from upcap.engine.precompute import compute_concepts_local_feat
from upcap.engine.precompute import compute_concepts_global_feat
from upcap.engine.precompute import compute_ref_text_feats_by_image_ids
from upcap.model.divider import Divider
from utils.dataset import CocoDataset, DType
from utils.logger import logger
from utils.dist import dist_startup

class _UniqueImageDataset(Dataset):
	def __init__(self, unique_images: list[tuple[str, Path]]):
		self.unique_images = unique_images

	def __len__(self) -> int:
		return len(self.unique_images)

	def __getitem__(self, index: int):
		image_id, image_path = self.unique_images[index]
		return {
			'image_id': image_id,
			'image': str(image_path),
		}

def store_concepts_local_feat():
	
	with logger('divider', 'loading', Cfg.is_master):
		divider = Divider()

	with logger('clip', 'loading', Cfg.is_master):
		clip_model, _ = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()

	with logger('dataset', 'loading', Cfg.is_master):
		dataset = CocoDataset(
			annotations = Cfg.coco_train_ann,
			images_path = Cfg.coco_train_image,
			cache_path = Cfg.coco_train_cache,
			dtype = DType.IMAGE
		)
		dataset.subset(65536)

	output_file = Cfg.concepts_local_feat_path
	temp_dir = output_file.parent / f'temp_parts_{output_file.stem}'
	
	if Cfg.is_master:
		output_file.parent.mkdir(parents=True, exist_ok=True)
		temp_dir.mkdir(parents=True, exist_ok=True)
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	with logger('upcap', 'computing features', Cfg.is_master):
		local_feats = compute_concepts_local_feat(
			dataset,
			divider,
			clip_model,
			batch_size=128,
			flatten=True,
		)
	
	part_path = temp_dir / f'part_{Cfg.rank}.pt'
	torch.save(local_feats, part_path)
	
	del local_feats
	del clip_model
	del divider
	gc.collect()
	torch.cuda.empty_cache()
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	if Cfg.is_master:
		with logger('upcap', 'merging features'):
			all_parts = []
			for rank in range(dist.get_world_size()):
				part_file = temp_dir / f'part_{rank}.pt'
				if part_file.exists():
					part_tensor = torch.load(
						part_file,
						map_location='cpu',
						weights_only=True
					)
					if part_tensor.numel() > 0:
						all_parts.append(part_tensor)
			
			if all_parts:
				final_tensor = torch.cat(all_parts, dim=0)
				torch.save(final_tensor, output_file)
				print(f"Total concept features saved: {final_tensor.shape}")
			else:
				print("No features extracted.")

def store_benchmark_precomputed_concepts():

	with logger('divider', 'loading', Cfg.is_master):
		divider = Divider()

	with logger('clip', 'loading', Cfg.is_master):
		clip_model, preprocess = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()

	with logger('dataset', 'loading', Cfg.is_master):
		base_dataset = CocoDataset(
			annotations=Cfg.coco_val_ann,
			images_path=Cfg.coco_val_image,
			cache_path=Cfg.coco_val_cache,
			dtype=DType.TEXT | DType.IMAGE,
		)

		image_id_to_refs: dict[str, list[str]] = {}
		seen_ids: set[str] = set()
		unique_images: list[tuple[str, Path]] = []
		for i in range(len(base_dataset)):
			item = base_dataset[i]
			image_path = Path(item['image'])
			image_id = image_path.name
			caption_text = item['text']
			if image_id not in image_id_to_refs:
				image_id_to_refs[image_id] = []
			image_id_to_refs[image_id].append(caption_text)
			if image_id in seen_ids:
				continue
			seen_ids.add(image_id)
			unique_images.append((image_id, image_path))

		dataset = _UniqueImageDataset(unique_images)

	output_file = Cfg.benchmark_precomputed_concepts_path
	temp_dir = output_file.parent / f'temp_parts_{output_file.stem}'

	if Cfg.is_master:
		output_file.parent.mkdir(parents=True, exist_ok=True)
		temp_dir.mkdir(parents=True, exist_ok=True)

	dist.barrier(device_ids=[torch.cuda.current_device()])

	with logger('upcap', 'computing benchmark precomputed concepts', Cfg.is_master):
		world_size = dist.get_world_size()
		local_unique_images = unique_images[Cfg.rank::world_size]
		local_image_ids = [img_id for img_id, _ in local_unique_images]
		local_dataset = _UniqueImageDataset(local_unique_images)

		global_feats = compute_concepts_global_feat(
			local_dataset,
			clip_model,
			preprocess,
			batch_size=512,
			use_distributed_sampler=False,
		)
		local_feats_by_image = compute_concepts_local_feat(
			local_dataset,
			divider,
			clip_model,
			batch_size=128,
			flatten=False,
			use_distributed_sampler=False,
		)

		assert isinstance(local_feats_by_image, list)

		zero_tokens = torch.zeros((1, 77), dtype=torch.long, device=Cfg.device)
		pad_feat = clip_model.encode_text(zero_tokens).float()
		pad_feat /= pad_feat.norm(dim=-1, keepdim=True)
		pad_feat = pad_feat.cpu().half()

		max_concepts = Cfg.max_concepts
		concepts_list: list[torch.Tensor] = []
		for global_feat, local_feat in zip(global_feats, local_feats_by_image):
			global_feat = global_feat.unsqueeze(0)
			if local_feat.numel() > 0:
				sim = (local_feat.float() @ global_feat.float().transpose(-2, -1)).squeeze(-1)
				local_feat = local_feat[sim.argsort(descending=True)]
				local_feat = local_feat[:max_concepts - 1]
				combined = torch.cat([global_feat, local_feat], dim=0)
			else:
				combined = global_feat

			pad_len = max_concepts - combined.shape[0]
			if pad_len > 0:
				combined = torch.cat([combined, pad_feat.expand(pad_len, -1)], dim=0)

			concepts_list.append(combined)

		concepts = torch.stack(concepts_list, dim=0)
		if concepts.shape[0] != len(local_image_ids):
			raise RuntimeError(
				f'Concept/Image ID length mismatch on rank {Cfg.rank}: '
				f'concepts={concepts.shape[0]}, image_ids={len(local_image_ids)}'
			)

		ref_text_feats = compute_ref_text_feats_by_image_ids(
			image_ids=local_image_ids,
			image_id_to_refs=image_id_to_refs,
			clip_model=clip_model,
			batch_size=4096,
		)
		if len(ref_text_feats) != len(local_image_ids):
			raise RuntimeError(
				f'Reference/Image ID length mismatch on rank {Cfg.rank}: '
				f'ref_text_feats={len(ref_text_feats)}, image_ids={len(local_image_ids)}'
			)


	part_path = temp_dir / f'part_{Cfg.rank}.pt'
	torch.save(
		{
			'image_ids': local_image_ids,
			'concepts': concepts,
			'ref_text_feats': ref_text_feats,
		},
		part_path
	)

	del global_feats
	del local_feats_by_image
	del concepts
	del clip_model
	del divider
	gc.collect()
	torch.cuda.empty_cache()

	dist.barrier(device_ids=[torch.cuda.current_device()])

	if Cfg.is_master:
		with logger('upcap', 'merging benchmark precomputed concepts'):
			all_ids: list[str] = []
			all_concepts: list[torch.Tensor] = []
			all_ref_text_feats: list[torch.Tensor] = []
			for rank in range(world_size):
				part_file = temp_dir / f'part_{rank}.pt'
				if part_file.exists():
					part = torch.load(part_file, map_location='cpu', weights_only=True)
					part_ids = part['image_ids']
					part_concepts = part['concepts']
					part_ref_text_feats = part['ref_text_feats']
					if len(part_ids) > 0 and part_concepts.numel() > 0:
						all_ids.extend(part_ids)
						all_concepts.append(part_concepts)
						all_ref_text_feats.extend(part_ref_text_feats)

			if all_concepts:
				final_concepts = torch.cat(all_concepts, dim=0)
				torch.save(
					{
						'image_ids': all_ids,
						'concepts': final_concepts,
						'ref_text_feats': all_ref_text_feats,
					},
					output_file,
				)
				print(f"Benchmark precomputed concepts saved: ids={len(all_ids)}, concepts={final_concepts.shape}")
			else:
				print("No benchmark precomputed concepts extracted.")
def store_concepts_global_feat():
	
	with logger('clip', 'loading', Cfg.is_master):
		clip_model, preprocess = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()

	with logger('dataset', 'loading', Cfg.is_master):
		dataset = CocoDataset(
			annotations = Cfg.coco_train_ann,
			images_path = Cfg.coco_train_image,
			cache_path = Cfg.coco_train_cache,
			dtype = DType.IMAGE
		)

	output_file = Cfg.concepts_global_feat_path
	temp_dir = output_file.parent / f'temp_parts_{output_file.stem}'
	
	if Cfg.is_master:
		output_file.parent.mkdir(parents=True, exist_ok=True)
		temp_dir.mkdir(parents=True, exist_ok=True)
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	with logger('upcap', 'computing global features', Cfg.is_master):
		global_feats = compute_concepts_global_feat(
			dataset,
			clip_model,
			preprocess,
			batch_size=512
		)
	
	part_path = temp_dir / f'part_{Cfg.rank}.pt'
	torch.save(global_feats, part_path)
	
	del global_feats
	del clip_model
	gc.collect()
	torch.cuda.empty_cache()
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	if Cfg.is_master:
		with logger('upcap', 'merging global features'):
			all_parts = []
			for rank in range(dist.get_world_size()):
				part_file = temp_dir / f'part_{rank}.pt'
				if part_file.exists():
					part_tensor = torch.load(
						part_file,
						map_location='cpu',
						weights_only=True
					)
					if part_tensor.numel() > 0:
						all_parts.append(part_tensor)
			
			if all_parts:
				final_tensor = torch.cat(all_parts, dim=0)
				torch.save(final_tensor, output_file)
				print(f"Total global features saved: {final_tensor.shape}")
			else:
				print("No global features extracted.")

if __name__ == '__main__':
	
	with dist_startup():
		store_benchmark_precomputed_concepts()
		# store_concepts_local_feat()
		# dist.barrier(device_ids=[torch.cuda.current_device()])
		# store_concepts_global_feat()