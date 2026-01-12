from utils.config import Config as ConfigBase

class _Model:
	latent_dim: int = 768
	latent_seq_len: int = 10
	adapter_depth: int = 4
	adapter_heads: int = 8
	text_mask_ratio: float = 0.3
	
	kl_weight: float = 0.01
	adv_weight: float = 0.1
	clip_loss_weight: float = 1.0
	
	max_seq_length: int = 30

class _Schedule:
	discriminator_learning_rate: float = 1e-4
	warmup_epochs: int = 5
	factor: int = 10
	batch_size: int = 32 * factor
	learning_rate: float = 1e-5 * factor
	warmup_steps: int = 1000 // factor

	micro_steps_iter: list[int] = [1, 4, 16]

class Config(ConfigBase, _Model, _Schedule):
	pass

class Cfg(Config):
	pass