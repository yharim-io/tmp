from utils.config import Config as ConfigBase

class _Model:
	prefix_length: int = 10
	num_layers: int = 8

class _Schedule:
	factor: int = 6
	batch_size: int = 40 * factor
	learning_rate: float = 2e-5 * factor
	warmup_steps: int = 5000 // factor

class Config(ConfigBase, _Model, _Schedule):
	pass

class Cfg(Config):
	pass
