from utils.dataset import CocoDataset, DType
from yottacap.engine.train import train
from yottacap.config import Cfg

dataset = CocoDataset(
	annotations = Cfg.coco_train_ann,
	images_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	dtype = DType.TEXT_EMB | DType.TEXT_FEAT | DType.IMAGE_EMB | DType.IMAGE_FEAT,
)

dataset.subset(4096)

train(
	dataset,
	output_dir = Cfg.root/'data/yottacap/coco/',
	epochs = 10,
    start_epoch = 0,
)