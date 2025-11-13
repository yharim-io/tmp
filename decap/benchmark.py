import torch
from torch import Tensor
import clip
from pathlib import Path
from tqdm import tqdm
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose

from decap.config import Cfg
from decap.layer.decap import DeCap
from decap.engine.decode import image_to_text, calc_text_features
from utils.dataset import CocoDataset, DType
from utils.metric import MetricEvaluator

def get_text_feat(feat_file: Path) -> Tensor:
	if feat_file.exists():
		text_features = torch.load(feat_file, weights_only=True).to(Cfg.device)
	else:
		ds_train = CocoDataset(
			annotations=Cfg.coco_train_ann,
			images_path=Cfg.coco_train_image,
			cache_path=Cfg.coco_train_cache,
			dtype=DType.TEXT,
			clip_model=clip_model,
			preprocess=preprocess
		)
		text_features = calc_text_features(clip_model, ds_train).to(Cfg.device)
		torch.save(text_features, feat_file)
	return text_features

def run_model(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	decap_model: DeCap,
	text_features: Tensor
) -> tuple[dict, dict]:
	
	dataset = CocoDataset(
		annotations=Cfg.coco_val_ann,
		images_path=Cfg.coco_val_image,
		cache_path=Cfg.coco_val_cache,
		dtype=DType.TEXT | DType.IMAGE,
		clip_model=clip_model,
		preprocess=preprocess
	)
	
	dataset.subset(100)

	ground_truth_annotations: dict = {}
	model_predictions: dict = {}

	for i in tqdm(range(len(dataset)), desc="Evaluating"):
		item = dataset[i]
		image_path = Path(item['image'])
		image_id = str(image_path.name)
		caption_text = item['text']

		if image_id not in ground_truth_annotations:
			ground_truth_annotations[image_id] = []
		ground_truth_annotations[image_id].append(caption_text)

		if image_id not in model_predictions:
			text = image_to_text(
				clip_model=clip_model,
				preprocess=preprocess,
				tokenizer=tokenizer,
				decap_model=decap_model,
				text_features=text_features,
				image_path=image_path
			)
			model_predictions[image_id] = [text]

	return ground_truth_annotations, model_predictions

if __name__ == '__main__':
	clip_model, preprocess = clip.load(Cfg.clip_pretrained_path, device=Cfg.device, jit=False)
	tokenizer = SimpleTokenizer()

	decap_model = DeCap().to(Cfg.device)
	decap_model.load_state_dict(
		torch.load(Cfg.root/'data/decap/text_image/coco/009.pt', map_location=Cfg.device, weights_only=True)
	)
	decap_model.eval()
	
	text_features = get_text_feat(Cfg.root/'data/decap/text_image/coco/train_text_features.pt')
	
	ground_truths, predictions = run_model(
		clip_model, preprocess, tokenizer, decap_model, text_features
	)

	metric_evaluator = MetricEvaluator(clip_model, preprocess, tokenizer)
	scores = metric_evaluator.compute(ground_truths, predictions)

	print(f"BLEU@4: {scores['Bleu_4']:.4f}")
	print(f"METEOR: {scores['METEOR']:.4f}")
	print(f"CIDEr:  {scores['CIDEr']:.4f}")
	print(f"SPICE:  {scores['SPICE']:.4f}")