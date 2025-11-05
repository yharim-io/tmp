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
	prefix_length: int = 10
	clip_model_type: str = 'ViT-B/32'
	clip_dim: int = 512
	max_seq_length: int = 25
	eos_token_id: int = 49407
	num_layers: int = 8
	
	# schedule
	factor: int = 16
	batch_size: int = 40 * factor
	learning_rate: float = 2e-5 * factor
	warmup_steps: int = 5000 // factor
	
	# dist
	rank: int = int(os.environ.get('LOCAL_RANK', 0))
	is_master: bool = rank == 0
	device: torch.device = torch.device(f'cuda:{rank}')