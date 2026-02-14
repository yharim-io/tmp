from upcap.engine.collate import CollateFn, GlobalCollateFn
from upcap.engine.train import train
from upcap.config import Cfg
from utils.dataset import CocoDataset, DType

if __name__ == '__main__':
	
	dataset = CocoDataset(
		annotations = Cfg.coco_train_ann,
		images_path = None,
		cache_path = None,
		dtype = DType.TEXT
	)
	
	collate_fn = CollateFn()
	# collate_fn = GlobalCollateFn()

	train(
		dataset,
		collate_fn = collate_fn,
		output_dir = Cfg.root/'data/upcap/coco',
		epochs = 50,
		# start_epoch = 10,
		# init_weights = Cfg.root/'data/upcap/coco/009.pt',
	)
