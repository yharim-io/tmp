import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from decap.config import Cfg
from dataset import CocoDataset
from decap.layer.decap import DeCap
from decap.engine.decode import get_text_features, image_to_text

feat_file = Cfg.root/'data/decap/coco/text_features.pt'

if feat_file.exists():
	text_features = torch.load(feat_file, weights_only=True)
else:
	dataset = CocoDataset(Cfg.coco_val_data)
	text_features = get_text_features(dataset)
	torch.save(text_features, feat_file)

print('[clip] loading...')
device = torch.device('cuda')
clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
tokenizer = SimpleTokenizer()
print('[clip] done')

decap_model = DeCap()
decap_model.load_state_dict(
	torch.load(
		Cfg.root/'data/decap/coco/000.pt',
		map_location=torch.device('cpu'),
		weights_only=True
	)
)

decap_model = decap_model.to('cuda')

for i in range(1, 9):
	
	text = image_to_text(
		clip_model=clip_model,
		preprocess=preprocess,
		tokenizer=tokenizer,
		decap_model=decap_model,
		text_features=text_features,
		image_path=Cfg.root/f'data/images/{i}.jpg'
	)

	print(text)

