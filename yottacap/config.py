from utils.config import Config as ConfigBase

class _Model:
	latent_dim: int = 768
	latent_seq_len: int = 4
	adapter_depth: int = 4
	adapter_heads: int = 8
	text_mask_ratio: float = 0.2
	
	adv_weight: float = 0.1
	
	max_seq_length: int = 30

class _Schedule:
	discriminator_learning_rate: float = 1e-4
	warmup_epochs: int = 5
	factor: int = 12
	batch_size: int = 32 * factor
	learning_rate: float = 1e-5 * factor

	micro_steps_iter: list[int] = [1, 1, 8]

class Config(ConfigBase, _Model, _Schedule):
	pass

class Cfg(Config):
	pass