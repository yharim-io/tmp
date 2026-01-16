import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer
from pathlib import Path

from yottacap.config import Cfg
from yottacap.layer.yottacap import YottaCap
from yottacap.engine.decode import image_to_text
from utils.logger import logger

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=torch.device('cuda'),
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('yottacap', 'loading'):
	yottacap_model = YottaCap()
	weights = Cfg.root/'data/yottacap/coco/epoch_49.pt'
	if weights.exists():
		yottacap_model.load_state_dict(
			torch.load(weights, map_location='cpu', weights_only=True)
		)
	yottacap_model = yottacap_model.to('cuda')
	yottacap_model.eval()

for i in range(1, 8):
	image_path = Cfg.root / f'data/example/{i}.jpg'
	if image_path.exists():
		text = image_to_text(
			preprocess=preprocess,
			tokenizer=tokenizer,
			model=yottacap_model,
			image_path=image_path
		)
		print(f"Image {i}: {text}")