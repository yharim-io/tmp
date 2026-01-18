from utils.config import Config as ConfigBase
from pathlib import Path

class _Path:
	concepts_image_path: Path = ConfigBase.root / 'data/upcap/concepts_image.pt'
	concepts_feat_path: Path = ConfigBase.root / 'data/upcap/concepts_clip.pt'
	upcap_output_dir: Path = ConfigBase.root / 'data/upcap/checkpoints/'

class _Param:
	learning_rate: float = 1e-4
	batch_size: int = 64
	max_concepts: int = 10

class Config(ConfigBase, _Path, _Param):
	pass

class Cfg(Config):
	pass