import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from decap.config import Cfg
from utils.coco import CocoDataset
from decap.layer.decap import DeCap
from decap.engine.decode import get_text_features, image_to_text

print('[clip] loading...')
clip_model, preprocess = clip.load(
	name=Cfg.clip_model_type,
	device=torch.device('cuda'),
	jit=False
)
clip_model.eval()
tokenizer = SimpleTokenizer()
print('[clip] done')

print('[decap] loading...')
feat_file = Cfg.root/'data/decap/coco/text_features.pt'
if feat_file.exists():
	text_features = torch.load(feat_file, weights_only=True)
else:
	print('[decap] calculating text features...')
	dataset = CocoDataset(Cfg.coco_val_data)
	text_features = get_text_features(dataset)
	torch.save(text_features, feat_file)
	print('[decap] calculating text features done')

decap_model = DeCap()
decap_model.load_state_dict(
	torch.load(
		Cfg.root/'data/decap/coco/000.pt',
		map_location=torch.device('cpu'),
		weights_only=True
	)
)
decap_model = decap_model.to('cuda')
decap_model.eval()
print('[decap] done')

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

