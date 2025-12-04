import torch
import clip
from torch import Tensor
from pathlib import Path
from tqdm import tqdm
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose

from yottacap.config import Cfg
from yottacap.layer.yottacap import YottaCap
from yottacap.engine.decode import calc_text_features
from yottacap.engine.decode_batch import image_to_text_batch
from utils.dataset import CocoDataset, DType
from utils.metric import MetricEvaluator
from utils.logger import logger

DATA_SPACE = Cfg.root/'data/yottacap/coco'
MODEL_WEIGHTS = DATA_SPACE / '049.pt'

def get_text_feat(feat_file: Path, clip_model: CLIP, preprocess: Compose) -> Tensor:
	if feat_file.exists():
		text_features = torch.load(feat_file, weights_only=True).to(Cfg.device)
	else:
		print('[yottacap] calculating text features...')
		dataset = CocoDataset(
			annotations = Cfg.coco_train_ann,
			images_path = Cfg.coco_train_image,
			cache_path = Cfg.coco_train_cache,
			dtype = DType.TEXT,
			clip_model = clip_model,
			preprocess = preprocess
		)
		text_features = calc_text_features(clip_model, dataset)
		torch.save(text_features, feat_file)
	return text_features

def run_model(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	text_features: Tensor,
	cache_path: Path | None = None,
	use_cache: bool = True,
	batch_size: int = 256
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
			tokenizer=tokenizer,
			yottacap_model=yottacap_model,
			text_features=text_features,
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
	
	with logger('yottacap', 'loading'):
		yottacap_model = YottaCap().to(Cfg.device)
		if MODEL_WEIGHTS.exists():
			yottacap_model.load_state_dict(
				torch.load(
					MODEL_WEIGHTS,
					map_location=Cfg.device,
					weights_only=True
				)
			)
		yottacap_model.eval()
		text_features = get_text_feat(DATA_SPACE/'train_text_features.pt', clip_model, preprocess)
	
	with logger('yottacap', 'running'):
		ground_truths, predictions = run_model(
			clip_model, preprocess, tokenizer, yottacap_model, text_features,
			cache_path=DATA_SPACE/'run_model.pt',
			use_cache=False
		)
	
	with logger('yottacap', 'evaluating'):
		metric_evaluator = MetricEvaluator(clip_model, preprocess, tokenizer)
		scores = metric_evaluator.compute(ground_truths, predictions)

	print(scores)