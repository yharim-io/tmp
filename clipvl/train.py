from utils.dataset import CocoDataset, DType
from clipvl.layer.clipvl import MappingType
from clipvl.engine.train import train
from clipvl.config import Cfg

MAPPING_TYPE = MappingType.Transformer

dataset = CocoDataset(
	annotation = Cfg.coco_train_ann,
	image_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	data_type = DType.TEXT_EMB | DType.IMAGE_FEAT
)

clipvl_model = train(
	dataset,
	output_dir = Cfg.root/f'data/clipvl/text_image/{MAPPING_TYPE.value}/coco/',
	epochs = 50,
	start_epoch = 0,
	mapping_type = MAPPING_TYPE,
	# init_weights = Cfg.root/f'data/clipvl/text_image/{MAPPING_TYPE.value}/coco/005.pt',
	text_only = False
)
