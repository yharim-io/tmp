from utils.dataset import CocoDataset, DType
from decap.engine.train import train
from decap.config import Cfg

dataset = CocoDataset(
	annotations = Cfg.coco_train_ann,
	images_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	dtype = DType.TEXT_EMB | DType.IMAGE_FEAT
)

decap_model = train(
	dataset,
	output_dir = Cfg.root/'data/decap/text_image/coco/',
	epochs = 50,
	start_epoch = 10,
	text_only = False,
	# init_weights = Config.root/'data/decap/coco/009.pt'
)
