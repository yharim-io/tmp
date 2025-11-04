from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class _Path:
	root: Path = Path('/home/yharim/Source/YottaCap/')
	
	@property
	def coco_train_data(self) -> Path:
		return self.root/'data/coco/annotations/captions_train2014.json'
	
	@property
	def coco_val_data(self) -> Path:
		return self.root/'data/coco/annotations/captions_val2014.json'
	
	@property
	def gpt2_config_path(self) -> Path:
		return self.root/'data/gpt2/decoder_config.pkl'

@dataclass
class _Model:
	max_seq_length: int = 25
	clip_model_type: str = 'ViT-B/32'
	clip_dim: int = 512
	eos_token_id: int = 49407

@dataclass
class _Policy:
	batch_size: int = 64
	learning_rate: float = 1e-5
	warmup_steps: int = 1000

class Config:
	path = _Path()
	model = _Model()
	policy = _Policy()
	
	rank: int = int(os.environ.get('LOCAL_RANK', -1))
	is_master: bool = rank == 0