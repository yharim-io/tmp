from utils.config import Config as ConfigBase
from pathlib import Path

class _Path:
	concepts_image_path: Path = ConfigBase.root / 'data/upcap/concepts_image.pt'
	concepts_feat_path: Path = ConfigBase.root / 'data/upcap/concepts_feat.pt'

class _Param:
	factor: int = 8
	batch_size: int = 64 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor
	
	max_concepts: int = 10

class Config(ConfigBase, _Path, _Param):
	pass

class Cfg(Config):
	pass