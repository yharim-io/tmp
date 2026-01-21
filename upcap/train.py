from upcap.config import Cfg
from utils.dataset import CocoDataset, DType
from upcap.engine.train import train
from upcap.engine.collate import CollateFn

if __name__ == '__main__':
	
	dataset = CocoDataset(
		annotations = Cfg.coco_train_ann,
		images_path = None,
		cache_path = None,
		dtype = DType.TEXT
	)
	
	collate_fn = CollateFn()

	train(
		dataset,
		collate_fn,
		output_dir = Cfg.root/'data/upcap/coco',
		epochs = 50,
		# start_epoch = 1,
		# init_weights = Cfg.root/'data/upcap/coco/000.pt'
	)