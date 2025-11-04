from dataset import CocoDataset
from decap.engine import train
from decap.config import Config

dataset = CocoDataset(Config.path.coco_train_data)

decap_model = train(
	dataset,
	output_dir=Config.path.root/'data/decap/coco/',
	log_dir=Config.path.root/'data/decap/coco/log/',
	epochs=50,
	start_epoch=10,
	init_weights=Config.path.root/'data/decap/coco/009.pt'
)