import torch
from torch import Tensor
from upcap.config import Cfg
from utils.dataset import CocoDataset, DType
from upcap.engine.train import train

if __name__ == '__main__':
	
	dataset = CocoDataset(
		annotations = Cfg.coco_train_ann,
		images_path = None,
		cache_path = None,
		dtype = DType.TEXT
	)

	train(
		dataset,
		output_dir = Cfg.root/'data/upcap/coco',
		epochs = 50,
		start_epoch = 0
	)