import torch
import torch.distributed as dist
import clip
from pathlib import Path
import os
import numpy as np
from PIL import Image
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from torch.utils.data import Dataset as TorchDataset, DataLoader

from upcap.config import Cfg
from upcap.model.upcap import UpCap
from upcap.model.divider import Divider
from upcap.engine.decode_batch import image_tensor_to_text_batch
from utils.dataset import Dataset, CocoDataset, DType
from utils.metric import MetricEvaluator
from utils.dist import dist_startup
from utils.logger import logger
from utils.tool import tqdm

DATASPACE = Cfg.root/'data/upcap/coco'
MODEL_WEIGHTS = DATASPACE/'009.pt'
CACHE_PATH = DATASPACE/'run_model_009.pt'
GLOBAL_ATTN = False
LOCAL_ATTN = False

class _UniqueImageDataset(TorchDataset):
	def __init__(self, unique_images: list[tuple[str, Path]], preprocess: Compose):
		self.unique_images = unique_images
		self.preprocess = preprocess

	def __len__(self) -> int:
		return len(self.unique_images)

	def __getitem__(self, index: int):
		image_id, image_path = self.unique_images[index]
		image = Image.open(image_path).convert('RGB')
		image_rgb = np.array(image)
		image_tensor = self.preprocess(image)
		return image_id, str(image_path), image_tensor, image_rgb

def _collate_unique_images(batch):
	batch_ids = [item[0] for item in batch]
	batch_paths = [Path(item[1]) for item in batch]
	image_tensor = torch.stack([item[2] for item in batch], dim=0)
	image_rgbs = [item[3] for item in batch]
	return batch_ids, batch_paths, image_tensor, image_rgbs

def run_model(
	dataset: Dataset,
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	upcap_model: UpCap,
	divider: Divider,
	cache_path: Path | None = None,
	use_cache: bool = True,
	batch_size: int = 64
) -> tuple[dict, dict]:
	
	if use_cache and cache_path is not None and cache_path.exists():
		if Cfg.is_master:
			print(f'Loading results from {cache_path}')
		return torch.load(cache_path, weights_only=True)
	
	ground_truth_annotations: dict = {}
	model_predictions: dict = {}
	
	unique_images = []
	seen_ids = set()

	for i in range(len(dataset)):
		item = dataset[i]
		image_path = Path(item['image'])
		image_id = str(image_path.name)
		caption_text = item['text']

		if image_id not in ground_truth_annotations:
			ground_truth_annotations[image_id] = []
		ground_truth_annotations[image_id].append(caption_text)

		if image_id not in seen_ids:
			unique_images.append((image_id, image_path))
			seen_ids.add(image_id)

	# total_images = len(unique_images)
	# for i in tqdm(range(0, total_images, batch_size), desc="Running Inference"):
	# 	batch_data = unique_images[i : i + batch_size]

	world_size = dist.get_world_size()
	local_images = unique_images[Cfg.rank::world_size]

	num_workers = min(8, max(1, (os.cpu_count() or 1) // max(1, world_size)))
	pin_memory = Cfg.device.type == 'cuda'

	loader_kwargs = {
		'batch_size': batch_size,
		'shuffle': False,
		'num_workers': num_workers,
		'pin_memory': pin_memory,
		'collate_fn': _collate_unique_images,
	}
	if num_workers > 0:
		loader_kwargs['prefetch_factor'] = 2
		loader_kwargs['persistent_workers'] = True

	local_dataset = _UniqueImageDataset(local_images, preprocess)
	local_loader = DataLoader(local_dataset, **loader_kwargs)

	iterator = local_loader
	if Cfg.is_master:
		iterator = tqdm(local_loader, desc='Running Inference', total=len(local_loader))

	with torch.inference_mode():
		for batch_ids, batch_paths, image_tensor, image_rgbs in iterator:
			batch_texts = image_tensor_to_text_batch(
				clip_model=clip_model,
				tokenizer=tokenizer,
				upcap_model=upcap_model,
				divider=divider,
				image_tensor=image_tensor,
				image_paths=batch_paths,
				image_rgbs=image_rgbs,
				global_attn=GLOBAL_ATTN,
				local_attn=LOCAL_ATTN,
			)

			for img_id, text in zip(batch_ids, batch_texts):
				model_predictions[img_id] = [text]

	# if cache_path is not None:
	# 	torch.save(
	# 		(ground_truth_annotations, model_predictions), cache_path)

	# return ground_truth_annotations, model_predictions

	temp_dir = DATASPACE / 'temp_results'
	os.makedirs(temp_dir, exist_ok=True)
	
	torch.save(model_predictions, temp_dir / f'part_{Cfg.rank}.pt')
	dist.barrier(device_ids=[torch.cuda.current_device()])
	
	if Cfg.is_master:
		full_predictions: dict = {}
		for r in range(world_size):
			part_file = temp_dir / f'part_{r}.pt'
			part_pred = torch.load(part_file, weights_only=True)
			full_predictions.update(part_pred)
			# os.remove(part_file)
		# os.rmdir(temp_dir)
		
		model_predictions = full_predictions

		if cache_path is not None:
			print(f"Saving full results to {cache_path}")
			torch.save(
				(ground_truth_annotations, model_predictions), cache_path)
	
	if not Cfg.is_master:
		ground_truth_annotations = {}
		model_predictions = {}

	return ground_truth_annotations, model_predictions

def merge_cache(
	dataset: Dataset,
	cache_path: Path,
) -> tuple[dict, dict]:
	
	ground_truth_annotations: dict = {}

	for i in range(len(dataset)):
		item = dataset[i]
		image_path = Path(item['image'])
		image_id = str(image_path.name)
		caption_text = item['text']

		if image_id not in ground_truth_annotations:
			ground_truth_annotations[image_id] = []
		ground_truth_annotations[image_id].append(caption_text)

	world_size = dist.get_world_size()
	temp_dir = DATASPACE / 'temp_results'

	if Cfg.is_master:
		full_predictions: dict = {}
		for r in range(world_size):
			part_file = temp_dir / f'part_{r}.pt'
			part_pred = torch.load(part_file, weights_only=True)
			full_predictions.update(part_pred)
			# os.remove(part_file)
		# os.rmdir(temp_dir)
		
		model_predictions = full_predictions

		if cache_path is not None:
			print(f"Saving full results to {cache_path}")
			torch.save(
				(ground_truth_annotations, model_predictions), cache_path)
	
	if not Cfg.is_master:
		ground_truth_annotations = {}
		model_predictions = {}

	return ground_truth_annotations, model_predictions

@dist_startup()
def main():

	with logger('clip', 'loading', Cfg.is_master):
		clip_model, preprocess = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()
		tokenizer = SimpleTokenizer()
	
	with logger('divider', 'loading', Cfg.is_master):
		divider = Divider()
	
	with logger('dataset', 'loading', Cfg.is_master):
		dataset = CocoDataset(
			annotations=Cfg.coco_val_ann,
			images_path=Cfg.coco_val_image,
			cache_path=Cfg.coco_val_cache,
			dtype=DType.TEXT | DType.IMAGE,
			clip_model=clip_model,
			preprocess=preprocess
		)
		# dataset.subset(1000)

	with logger('upcap', 'loading', Cfg.is_master):
		upcap_model = UpCap(
			enable_concepts_global_buffer=GLOBAL_ATTN,
			enable_concepts_local_buffer=LOCAL_ATTN,
		)
		static_dict = torch.load(
			MODEL_WEIGHTS,
			map_location=Cfg.device,
			weights_only=True
		)
		if any(k.startswith('_orig_mod.') for k in static_dict.keys()):
			static_dict = {
				(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
				for k, v in static_dict.items()
			}
		upcap_model.load_state_dict(static_dict)
		upcap_model = upcap_model.to(Cfg.device)
		upcap_model.eval()
	
	with logger('upcap', 'running', Cfg.is_master):
		ground_truths, predictions = run_model(
			dataset,
			clip_model, preprocess, tokenizer, upcap_model, divider,
			cache_path=CACHE_PATH,
			use_cache=True
		)
		# ground_truths, predictions = merge_cache(dataset, CACHE_PATH)

	if Cfg.is_master:
		with logger('upcap', 'evaluating'):
			metric_evaluator = MetricEvaluator(clip_model, preprocess, tokenizer)
			scores = metric_evaluator.compute(ground_truths, predictions)

		print(scores)

if __name__ == '__main__':
	
	main()
