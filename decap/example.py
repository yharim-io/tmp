import torch
from torch import Tensor
import clip
from clip.simple_tokenizer import SimpleTokenizer
from pathlib import Path

from decap.config import Cfg
from utils.dataset import CocoDataset, DType
from decap.layer.decap import DeCap
from decap.engine.decode import calc_text_features, image_to_text
from utils.logger import logger

def get_text_feat(feat_file: Path) -> Tensor:
	if feat_file.exists():
		text_features = torch.load(feat_file, weights_only=True)
	else:
		print('[decap] calculating text features...')
		dataset = CocoDataset(
			annotation = Cfg.coco_val_ann,
			cache_path = Cfg.coco_val_cache,
			data_type = DType.TEXT,
			clip_model = clip_model,
			preprocess = preprocess
		)
		text_features = calc_text_features(clip_model, dataset)
		torch.save(text_features, feat_file)
	return text_features

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=torch.device('cuda'),
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('decap', 'loading'):
	decap_model = DeCap()
	decap_model.load_state_dict(
		torch.load(
			Cfg.root/'data/decap/text_image/coco/002.pt',
			map_location=torch.device('cpu'),
			weights_only=True
		)
	)
	decap_model = decap_model.to('cuda')
	decap_model.eval()
	text_features = get_text_feat(Cfg.root/'data/decap/text_image/coco/text_features.pt')

for i in range(1, 9):
	
	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		decap_model = decap_model,
		text_features = text_features,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)

	print(text)

