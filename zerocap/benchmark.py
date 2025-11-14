import torch
import clip
from pathlib import Path
from tqdm import tqdm
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose

from zerocap.config import Cfg
from zerocap.layer.gpt2 import GPT2
from zerocap.engine.decode_batch import image_to_text_batch
from utils.dataset import CocoDataset, DType
from utils.metric import MetricEvaluator
from utils.logger import logger

DATA_SPACE = Cfg.root/'data/zerocap/coco'
DATA_SPACE.mkdir(parents=True, exist_ok=True)

def run_model(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	gpt_model: GPT2,
	cache_path: Path | None = None,
	use_cache: bool = True,
	batch_size: int = 32
) -> tuple[dict, dict]:
	
	if use_cache and cache_path is not None and cache_path.exists():
		print(f'Loading results from {cache_path}')
		return torch.load(cache_path, weights_only=True)
	
	dataset = CocoDataset(
		annotations=Cfg.coco_val_ann,
		images_path=Cfg.coco_val_image,
		cache_path=Cfg.coco_val_cache,
		dtype=DType.TEXT | DType.IMAGE,
		clip_model=clip_model,
		preprocess=preprocess
	)
	
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

	total_images = len(unique_images)
	for i in tqdm(range(0, total_images, batch_size), desc="Running Inference"):
		batch_data = unique_images[i : i + batch_size]
		batch_ids = [item[0] for item in batch_data]
		batch_paths = [item[1] for item in batch_data]

		batch_texts = image_to_text_batch(
			clip_model=clip_model,
			preprocess=preprocess,
			gpt_model=gpt_model,
			tokenizer=tokenizer,
			image_paths=batch_paths
		)

		for img_id, text in zip(batch_ids, batch_texts):
			model_predictions[img_id] = [text]

	if cache_path is not None:
		torch.save(
			(ground_truth_annotations, model_predictions), cache_path)

	return ground_truth_annotations, model_predictions

if __name__ == '__main__':
	
	with logger('clip', 'loading'):
		clip_model, preprocess = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()
		tokenizer = SimpleTokenizer()
	
	with logger('gpt2', 'loading'):
		gpt_model = GPT2().to(Cfg.device)
		gpt_model.eval()
	
	with logger('zerocap', 'running'):
		ground_truths, predictions = run_model(
			clip_model, preprocess, tokenizer, gpt_model,
			cache_path=DATA_SPACE/'run_model.pt',
			use_cache=False,
			batch_size=Cfg.batch_size
		)
	
	with logger('zerocap', 'evaluating'):
		metric_evaluator = MetricEvaluator(clip_model, preprocess, tokenizer)
		scores = metric_evaluator.compute(ground_truths, predictions)

	print(scores)