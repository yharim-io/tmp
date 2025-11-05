from pathlib import Path
import os
import torch

class Cfg:
	
	# path
	root: Path = Path('/home/yharim/Source/YottaCap/')
	coco_train_data = root/'data/coco/annotations/captions_train2014.json'
	coco_val_data = root/'data/coco/annotations/captions_val2014.json'
	coco_cache = root/'data/coco/cache.pkl'
	gpt2_config_path = root/'data/gpt2/decoder_config.pkl'
	
	# model
	max_seq_length: int = 25
	clip_model_type: str = 'ViT-B/32'
	clip_dim: int = 512
	eos_token_id: int = 49407
	
	# schedule
	factor: int = 16
	batch_size: int = 64 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor
	
	# dist
	rank: int = int(os.environ.get('LOCAL_RANK', 0))
	is_master: bool = rank == 0
	device: torch.device = torch.device(f'cuda:{rank}')
