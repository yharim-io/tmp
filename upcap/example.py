import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from upcap.config import Cfg
from upcap.model.upcap import UpCap
from upcap.model.divider import Divider
from upcap.engine.decode import image_to_text
from utils.logger import logger

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=Cfg.device,
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('divider', 'loading'):
	divider = Divider()

with logger('upcap', 'loading'):
	upcap_model = UpCap()
	static_dict = torch.load(
		Cfg.root/'data/upcap/coco/000.pt',
		map_location='cpu',
		weights_only=True
	)
	static_dict.pop('concepts_feat', None) # concepts_feat deprecated
	upcap_model.load_state_dict(static_dict)
	upcap_model = upcap_model.to(Cfg.device)
	upcap_model.eval()

for i in range(1, 9):

	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		upcap_model = upcap_model,
		divider = divider,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)

	print(text)
