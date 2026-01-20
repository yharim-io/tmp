from utils.config import Config as ConfigBase
from pathlib import Path

class _Path:
	concepts_local_image_path: Path = ConfigBase.root / 'data/upcap/concepts_local_image.pt'
	concepts_local_feat_path: Path = ConfigBase.root / 'data/upcap/concepts_local_feat.pt'
	concepts_global_feat_path: Path = ConfigBase.root / 'data/upcap/concepts_global_feat.pt'

class _Param:
	factor: int = 12
	batch_size: int = 64 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor
	
	max_concepts: int = 10

class Config(ConfigBase, _Path, _Param):
	pass

class Cfg(Config):
	pass