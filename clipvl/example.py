import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from clipvl.config import Cfg
from clipvl.layer.clipvl import ClipVLModel, MappingType
from clipvl.engine.decode import image_to_text

MAPPING_TYPE = MappingType.Transformer

print('[clip] loading...')
clip_model, preprocess = clip.load(
	name=Cfg.clip_pretrained_path,
	device=torch.device('cuda'),
	jit=False
)
clip_model.eval()
tokenizer = SimpleTokenizer()
print('[clip] loading done')

print('[clipvl] loading...')
clipvl_model = ClipVLModel(mapping_type = MAPPING_TYPE)
clipvl_model.load_state_dict(
	torch.load(
		Cfg.root/f'data/clipvl/text_image/{MAPPING_TYPE.value}/coco/000.pt',
		map_location=torch.device('cpu'),
		weights_only=True
	)
)
clipvl_model = clipvl_model.to('cuda')
clipvl_model.eval()
print('[clipvl] loading done')

for i in range(1, 9):

	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		clipvl_model = clipvl_model,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)

	print(text)
