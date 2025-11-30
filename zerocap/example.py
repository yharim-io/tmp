import clip

from zerocap.config import Cfg
from zerocap.layer.zerocap import ZeroCap
from zerocap.engine.decode import image_to_text
from utils.logger import logger

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=Cfg.device,
		jit=False
	)
	clip_model.eval()

with logger('zerocap', 'loading'):
	zerocap_model = ZeroCap(clip_model=clip_model).to(Cfg.device)
	zerocap_model.eval()

for i in range(1, 9):
	image_path = Cfg.root/f'data/example/{i}.jpg'
	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		zerocap_model = zerocap_model,
		image_path = image_path
	)
	print(f"Image {i}.jpg: {text}")