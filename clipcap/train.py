from dataset import CocoDataset
from clipcap.engine import train
from clipcap.config import Config
from clipcap.layer import MappingType

dataset = CocoDataset(Config.path.coco_train_data)

output_path = Config.path.root/'data/clipcap/coco/'
log_path = Config.path.root/'data/clipcap/coco/log/'

# init_weights_path = Config.path.root/'data/clipcap/coco/009.pt'
init_weights_path = None

model = train(
	dataset,
	output_dir=output_path,
	log_dir=log_path,
	epochs=50,
	start_epoch=0,
	init_weights=init_weights_path,
	mapping_type=MappingType.MLP, 
	only_prefix=False 
)
