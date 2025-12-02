import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from clipcap.config import Cfg
from clipcap.layer.clipcap import ClipCapModel, MappingType
from clipcap.engine.decode import image_to_text
from utils.logger import logger

MAPPING_TYPE = MappingType.Transformer

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=torch.device('cuda'),
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('clipcap', 'loading'):
	clipcap_model = ClipCapModel(mapping_type = MAPPING_TYPE)
	clipcap_model.load_state_dict(
		torch.load(
			Cfg.root/f'data/clipcap/text_image/{MAPPING_TYPE.value}/coco/001.pt',
			map_location=torch.device('cpu'),
			weights_only=True
		)
	)
	clipcap_model = clipcap_model.to('cuda')
	clipcap_model.eval()

for i in range(1, 9):

	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		clipcap_model = clipcap_model,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)

	print(text)
