from utils.dataset import CocoDataset, DType
from yottacap.engine.train import train
from yottacap.config import Cfg

dataset = CocoDataset(
	annotations = Cfg.coco_train_ann,
	images_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	dtype = DType.TEXT_EMB | DType.TEXT_FEAT | DType.IMAGE_EMB | DType.IMAGE_FEAT,
)

# dataset.subset(2048)

train(
	dataset,
	output_dir = Cfg.root/'data/yottacap/coco/',
	epochs = 50,
    start_epoch = 0,
    # init_weights = Cfg.root/'data/yottacap/coco/epoch_3.pt',
)