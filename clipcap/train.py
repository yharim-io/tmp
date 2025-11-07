from utils.dataset import CocoDataset, DType
from clipcap.layer.clipcap import MappingType
from clipcap.engine.train import train
from clipcap.config import Cfg

MAPPING_TYPE = MappingType.MLP

dataset = CocoDataset(
	annotations = Cfg.coco_train_ann,
	images_path = Cfg.coco_train_image,
	cache_path = Cfg.coco_train_cache,
	dtype = DType.TEXT_EMB | DType.IMAGE_FEAT
)

clipcap_model = train(
	dataset,
	output_dir = Cfg.root/f'data/clipcap/text_image/{MAPPING_TYPE.value}/coco/',
	epochs = 50,
	start_epoch = 0,
	mapping_type = MAPPING_TYPE,
	# init_weights = Cfg.root/f'data/clipcap/text_image/{MAPPING_TYPE.value}/coco/005.pt',
	text_only = False
)
