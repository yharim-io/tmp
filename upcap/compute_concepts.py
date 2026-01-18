import torch
import torch.distributed as dist
import clip
import os

from upcap.config import Cfg
from upcap.engine.compute_concepts import compute_concepts
from upcap.model.divider import Divider
from utils.dataset import CocoDataset, DType
from utils.logger import logger

if __name__ == '__main__':
	
	torch.cuda.set_device(Cfg.device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)
	
	with logger('clip', 'loading', Cfg.is_master):
		clip_model, preprocess = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()

	with logger('divider', 'loading', Cfg.is_master):
		divider = Divider()

	dataset = CocoDataset(
		annotations = Cfg.coco_train_ann,
		images_path = Cfg.coco_train_image,
		cache_path = Cfg.coco_train_cache,
		dtype = DType.IMAGE
	)
	
	output_file = Cfg.root / 'data/upcap/concepts.pt'
	temp_dir = output_file.parent / 'temp_parts'
	
	if Cfg.is_master:
		output_file.parent.mkdir(parents=True, exist_ok=True)
		temp_dir.mkdir(parents=True, exist_ok=True)
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	with logger('upcap', 'computing concepts', Cfg.is_master):
		local_concepts = compute_concepts(
			dataset,
			clip_model,
			preprocess,
			divider,
			batch_size=256
		)
	
	part_path = temp_dir / f'part_{Cfg.rank}.pt'
	torch.save(local_concepts, part_path)
	
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	if Cfg.is_master:
		with logger('upcap', 'merging concepts'):
			all_parts = []
			for rank in range(dist.get_world_size()):
				part_file = temp_dir / f'part_{rank}.pt'
				if part_file.exists():
					part_tensor = torch.load(part_file, weights_only=True)
					if part_tensor.numel() > 0:
						all_parts.append(part_tensor)
					os.remove(part_file)
			
			if all_parts:
				final_tensor = torch.cat(all_parts, dim=0)
				torch.save(final_tensor, output_file)
				print(f"Total concepts saved: {final_tensor.shape[0]}")
			else:
				print("No concepts extracted.")
			
			os.rmdir(temp_dir)