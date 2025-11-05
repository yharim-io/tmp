from dataset import CocoDataset
from decap.engine.train import train
from decap.config import Cfg

dataset = CocoDataset(
	train_data = Cfg.coco_train_data,
	cache_path = Cfg.coco_cache
)

decap_model = train(
	dataset,
	output_dir = Cfg.root/'data/tmp/coco/',
	epochs = 10,
	start_epoch = 0,
	# init_weights = Config.root/'data/decap/coco/009.pt'
)