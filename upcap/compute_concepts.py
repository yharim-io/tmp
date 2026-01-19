import torch
import torch.distributed as dist
import clip
from clip.model import CLIP
import gc

from upcap.config import Cfg
from upcap.engine.compute_concepts import compute_concepts_image, compute_concepts_feat
from upcap.model.divider import Divider
from utils.dataset import CocoDataset, DType
from utils.logger import logger
	
def store_concepts_image():
	
	with logger('divider', 'loading', Cfg.is_master):
		divider = Divider()

	with logger('dataset', 'loading', Cfg.is_master):
		dataset = CocoDataset(
			annotations = Cfg.coco_train_ann,
			images_path = Cfg.coco_train_image,
			cache_path = Cfg.coco_train_cache,
			dtype = DType.IMAGE
		)
		dataset.subset(4096)
	
	output_file = Cfg.concepts_image_path
	temp_dir = output_file.parent / 'temp_parts_image'
	
	if Cfg.is_master:
		output_file.parent.mkdir(parents=True, exist_ok=True)
		temp_dir.mkdir(parents=True, exist_ok=True)
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	with logger('upcap', 'extracting images', Cfg.is_master):
		local_concepts = compute_concepts_image(
			dataset,
			divider,
			batch_size=256
		)
	
	part_path = temp_dir / f'concepts_image_part_{Cfg.rank}.pt'
	torch.save(local_concepts, part_path)
	
	del local_concepts
	del divider
	gc.collect()
	torch.cuda.empty_cache()
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	if Cfg.is_master:
		with logger('upcap', 'merging images'):
			all_parts = []
			for rank in range(dist.get_world_size()):
				part_file = temp_dir / f'concepts_image_part_{rank}.pt'
				if part_file.exists():
					part_tensor = torch.load(
						part_file,
						map_location='cpu',
						weights_only=True
					)
					if part_tensor.numel() > 0:
						all_parts.append(part_tensor)
					# os.remove(part_file)
			
			if all_parts:
				final_tensor = torch.cat(all_parts, dim=0)
				torch.save(final_tensor, output_file)
				print(f"Total concept images saved: {final_tensor.shape[0]}")
			else:
				print("No concepts extracted.")
			
			# os.rmdir(temp_dir)

def store_concepts_feat():
	
	with logger('clip', 'loading', Cfg.is_master):
		clip_model, _ = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()

	output_file = Cfg.concepts_feat_path
	temp_dir = output_file.parent / 'temp_parts_feat'
	
	if Cfg.is_master:
		output_file.parent.mkdir(parents=True, exist_ok=True)
		temp_dir.mkdir(parents=True, exist_ok=True)
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	with logger('upcap', 'computing features', Cfg.is_master):
		local_feats = compute_concepts_feat(
			clip_model
		)
	
	part_path = temp_dir / f'concepts_feat_part_{Cfg.rank}.pt'
	torch.save(local_feats, part_path)
	
	del local_feats
	del clip_model
	gc.collect()
	torch.cuda.empty_cache()
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	if Cfg.is_master:
		with logger('upcap', 'merging features'):
			all_parts = []
			for rank in range(dist.get_world_size()):
				part_file = temp_dir / f'concepts_feat_part_{rank}.pt'
				if part_file.exists():
					part_tensor = torch.load(
						part_file,
						map_location='cpu',
						weights_only=True
					)
					if part_tensor.numel() > 0:
						all_parts.append(part_tensor)
					# os.remove(part_file)
			
			if all_parts:
				final_tensor = torch.cat(all_parts, dim=0)
				torch.save(final_tensor, output_file)
				print(f"Total concept features saved: {final_tensor.shape}")
			else:
				print("No features extracted.")
			
			# os.rmdir(temp_dir)

if __name__ == '__main__':
	
	torch.cuda.set_device(Cfg.device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.backends.cudnn.benchmark = True
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)

	try:
		store_concepts_image()
		dist.barrier(device_ids=[torch.cuda.current_device()])
		store_concepts_feat()
	finally:
		dist.destroy_process_group()