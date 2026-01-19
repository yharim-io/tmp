import torch
import torch.distributed as dist
import clip
from pathlib import Path
import os
os.environ["TQDM_NCOLS"] = "70"
from tqdm import tqdm
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose

from upcap.config import Cfg
from upcap.model.upcap import UpCap
from upcap.model.divider import Divider
from upcap.engine.decode_batch import image_to_text_batch
from utils.dataset import Dataset, CocoDataset, DType
from utils.metric import MetricEvaluator
from utils.logger import logger

DATASPACE = Cfg.root/'data/upcap/coco'
MODEL_WEIGHTS = DATASPACE/'013.pt'
CACHE_PATH = DATASPACE/'run_model_013.pt'

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

	iterator = range(0, len(local_images), batch_size)
	if Cfg.is_master: iterator = tqdm(iterator, desc="Running Inference")

	for i in iterator:
		batch_data = local_images[i : i + batch_size]

		batch_ids = [item[0] for item in batch_data]
		batch_paths = [item[1] for item in batch_data]

		batch_texts = image_to_text_batch(
			clip_model=clip_model,
			preprocess=preprocess,
			tokenizer=tokenizer,
			upcap_model=upcap_model,
			divider=divider,
			image_paths=batch_paths
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

if __name__ == '__main__':
	
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.cuda.set_device(Cfg.device)
	torch.manual_seed(42)
	torch.cuda.manual_seed(42)

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
		upcap_model = UpCap()
		upcap_model = upcap_model.to(Cfg.device)
		upcap_model.load_state_dict(
			torch.load(
				MODEL_WEIGHTS,
				map_location=Cfg.device,
				weights_only=True
			)
		)
		upcap_model.eval()
	
	with logger('upcap', 'running', Cfg.is_master):
		ground_truths, predictions = run_model(
			dataset,
			clip_model, preprocess, tokenizer, upcap_model, divider,
			cache_path=CACHE_PATH,
			use_cache=True
		)

	if Cfg.is_master:
		with logger('upcap', 'evaluating'):
			metric_evaluator = MetricEvaluator(clip_model, preprocess, tokenizer)
			scores = metric_evaluator.compute(ground_truths, predictions)

		print(scores)

	dist.destroy_process_group()
