from utils.config import Config as ConfigBase
from pathlib import Path

class _Path:
	concepts_local_feat_path: Path = ConfigBase.root / 'data/upcap/concepts_local_feat_65536.pt'
	concepts_global_feat_path: Path = ConfigBase.root / 'data/upcap/concepts_global_feat.pt'
	benchmark_precomputed_concepts_path: Path = ConfigBase.root / 'data/upcap/benchmark_precomputed_concepts.pt'
	noise_rf_path: Path = ConfigBase.root / 'data/upcap/noise_rf.pt'

class _Model:
	# https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
	clip_mean: list[float] = [0.48145466, 0.4578275, 0.40821073]
	clip_std: list[float] = [0.26862954, 0.26130258, 0.27577711]

class _Schedule:
	factor: int = 8
	batch_size: int = 64 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor
	
	max_concepts: int = 3

class Config(ConfigBase, _Path, _Model, _Schedule):
	pass

class Cfg(Config):
	pass