import torch
import clip
from upcap.config import Cfg
from upcap.engine.compute_concepts import compute_concepts
from upcap.model.divider import Divider
from utils.dataset import CocoDataset, DType
from utils.logger import logger

if __name__ == '__main__':
	
	with logger('clip', 'loading'):
		clip_model, preprocess = clip.load(
			Cfg.clip_pretrained_path,
			device=Cfg.device,
			jit=False
		)
		clip_model.eval()

	with logger('divider', 'loading'):
		divider = Divider()

	dataset = CocoDataset(
		annotations = Cfg.coco_train_ann,
		images_path = Cfg.coco_train_image,
		cache_path = Cfg.coco_train_cache,
		dtype = DType.IMAGE
	)
	
	output_file = Cfg.root / 'data/upcap/concepts.pt'
	output_file.parent.mkdir(parents=True, exist_ok=True)
	
	with logger('upcap', 'computing concepts'):
		concepts = compute_concepts(
			dataset,
			clip_model,
			preprocess,
			divider
		)
	
	with logger('upcap', 'saving concepts'):
		torch.save(concepts, output_file)