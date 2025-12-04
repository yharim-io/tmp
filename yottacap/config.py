from utils.config import Config as ConfigBase

class _Model:
	ln2: float = 0.69314718056
	asp_temperature: float = 0.01
	context_length: int = 77
	vocab_size: int = 49408

class _Schedule:
	factor: int = 8
	batch_size: int = 64 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor

class Config(ConfigBase, _Model, _Schedule):
	pass

class Cfg(Config):
	pass