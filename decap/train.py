from utils.coco import CocoDataset, DataType
from decap.engine.train import train
from decap.config import Cfg

dataset = CocoDataset(
	annotation = Cfg.coco_train_ann,
	image_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	data_type = DataType.TEXT_EMB | DataType.IMAGE_EMB | DataType.TEXT | DataType.IMAGE
)

decap_model = train(
	dataset,
	output_dir = Cfg.root/'data/decap/text_image/coco/',
	epochs = 50,
	start_epoch = 10,
	text_only = False,
	# init_weights = Config.root/'data/decap/coco/009.pt'
)
