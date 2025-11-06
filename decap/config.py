from utils.config import Config as BaseConfig

class _Schedule:
	factor: int = 8
	batch_size: int = 64 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor

class Config(BaseConfig, _Schedule):
	pass

class Cfg(Config):
	pass
