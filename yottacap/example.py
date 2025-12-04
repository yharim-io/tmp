import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer
from pathlib import Path
from torch import Tensor

from yottacap.config import Cfg
from yottacap.layer.yottacap import YottaCap
from yottacap.engine.decode import image_to_text, calc_text_features
from utils.dataset import CocoDataset, DType
from utils.logger import logger

def get_text_feat(feat_file: Path) -> Tensor:
	if feat_file.exists():
		text_features = torch.load(feat_file, weights_only=True).to(Cfg.device)
	else:
		print('[yottacap] calculating text features...')
		dataset = CocoDataset(
			annotations = Cfg.coco_val_ann,
			images_path = Cfg.coco_val_image,
			cache_path = Cfg.coco_val_cache,
			dtype = DType.TEXT,
			clip_model = clip_model,
			preprocess = preprocess
		)
		text_features = calc_text_features(clip_model, dataset)
		torch.save(text_features, feat_file)
	return text_features

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=Cfg.device,
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('yottacap', 'loading'):
	yottacap_model = YottaCap()
	weights_path = Cfg.root/'data/yottacap/coco/000.pt'
	if weights_path.exists():
		yottacap_model.load_state_dict(
			torch.load(weights_path, map_location='cpu', weights_only=True)
		)
	yottacap_model = yottacap_model.to(Cfg.device)
	yottacap_model.eval()
	text_features = get_text_feat(Cfg.root/'data/yottacap/coco/text_features.pt')

for i in range(1, 9):
	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		yottacap_model = yottacap_model,
		text_features = text_features,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)
	print(f"Image {i}: {text}")