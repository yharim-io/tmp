import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from clipcap.config import Cfg
from clipcap.layer.clipcap import ClipCaptionModel, MappingType
from clipcap.engine.decode import image_to_text

MAPPING_TYPE = MappingType.Transformer

print('[clip] loading...')
clip_model, preprocess = clip.load(
	name=Cfg.clip_model_type,
	device=torch.device('cuda'),
	jit=False
)
clip_model.eval()
tokenizer = SimpleTokenizer()
print('[clip] done')

print('[clipcap] loading...')
clipcap_model = ClipCaptionModel(
	prefix_length = Cfg.prefix_length,
	clip_length = Cfg.prefix_length,
	prefix_size = Cfg.clip_dim,
	num_layers = Cfg.num_layers,
	mapping_type = MAPPING_TYPE
)
clipcap_model.load_state_dict(
	torch.load(
		Cfg.root/f'data/clipcap/{MAPPING_TYPE}/coco/010.pt',
		map_location=torch.device('cpu'),
		weights_only=True
	)
)
clipcap_model = clipcap_model.to('cuda')
clipcap_model.eval()
print('[clipcap] done')

for i in range(1, 9):

	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		clipcap_model = clipcap_model,
		image_path = Cfg.root/f'data/example/{i}.jpg'
	)

	print(f"Image {i}.jpg: {text}")
