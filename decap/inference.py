import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from decap.config import Config
from dataset import CocoDataset
from decap.layer import DeCap
from decap.engine import get_text_features, image_to_text

feat_file = Config.path.root/'data/decap/coco/text_features.pt'

if feat_file.exists():
	text_features = torch.load(feat_file, weights_only=True)
else:
	dataset = CocoDataset(Config.path.coco_val_data)
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
		Config.path.root/'data/decap/coco/037.pt',
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
		image_path=Config.path.root/f'data/images/{i}.jpg'
	)

	print(text)

