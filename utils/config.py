from pathlib import Path
import os
import torch

class _Path:
	root: Path = Path('/home/yharim/Source/YottaCap/')
	coco_train_ann: Path = root/'data/coco/annotations/captions_train2014.json'
	coco_train_image: Path = root/'data/coco/train2014/'
	coco_train_cache: Path = root/'data/coco/train_cache.pkl'
	coco_val_ann: Path = root/'data/coco/annotations/captions_val2014.json'
	coco_val_image: Path = root/'data/coco/val2014/'
	coco_val_cache: Path = root/'data/coco/val_cache.pkl'
	gpt2_config_path: Path = root/'data/gpt2/decoder_config.pkl'
	clip_pretrained_path: Path = root/'data/clip/ViT-B-32.pt'

class _Dist:
	rank: int = int(os.environ.get('LOCAL_RANK', 0))
	is_master: bool = rank == 0
	device: torch.device = torch.device(f'cuda:{rank}')

class _Clip:
	clip_model_type: str = 'ViT-B/32'
	max_seq_length: int = 25
	clip_dim: int = 512
	eos_token_id: int = 49407

class Config(_Path, _Dist, _Clip):
	pass

