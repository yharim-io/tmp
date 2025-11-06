import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from decap.config import Cfg
from utils.coco import CocoDataset, DataType
from decap.layer.decap import DeCap
from decap.engine.decode import get_text_features, image_to_text

print('[clip] loading...')
clip_model, preprocess = clip.load(
	name=Cfg.clip_pretrained_path,
	device=torch.device('cuda'),
	jit=False
)
clip_model.eval()
tokenizer = SimpleTokenizer()
print('[clip] loading done')

print('[decap] loading...')
feat_file = Cfg.root/'data/decap/text_image/coco/text_features.pt'
if feat_file.exists():
	text_features = torch.load(feat_file, weights_only=True)
else:
	print('[decap] calculating text features...')
	dataset = CocoDataset(
		annotation = Cfg.coco_val_ann,
		cache_path = Cfg.coco_val_cache,
		data_type = DataType.TEXT
	)
	text_features = get_text_features(clip_model, dataset)
	torch.save(text_features, feat_file)
	print('[decap] calculating text features done')

decap_model = DeCap()
decap_model.load_state_dict(
	torch.load(
		Cfg.root/'data/decap/text_image/coco/009.pt',
		map_location=torch.device('cpu'),
		weights_only=True
	)
)
decap_model = decap_model.to('cuda')
decap_model.eval()
print('[decap] loading done')

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

