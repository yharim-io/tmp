from utils.coco import CocoDataset, DataType
from clipcap.layer.clipcap import MappingType
from clipcap.engine.train import train
from clipcap.config import Cfg

MAPPING_TYPE = MappingType.Transformer

dataset = CocoDataset(
	annotation = Cfg.coco_train_ann,
	image_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	data_type = DataType.TEXT_EMB
)

clipcap_model = train(
	dataset,
	output_dir = Cfg.root/f'data/clipcap/{MAPPING_TYPE.value}/coco/',
	epochs = 55,
	start_epoch = 6,
	mapping_type = MAPPING_TYPE,
	init_weights = Cfg.root/f'data/clipcap/{MAPPING_TYPE.value}/coco/005.pt',
)
