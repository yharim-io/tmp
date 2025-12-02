import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from yottacap.config import Cfg
from yottacap.layer.yottacap import YottaCap
from yottacap.engine.decode import image_to_text
from utils.logger import logger

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

for i in range(1, 9):
	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		yottacap_model = yottacap_model,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)
	print(f"Image {i}: {text}")