from utils.config import Config as ConfigBase

class _Model:
	forbidden_tokens_path: str = ConfigBase.root/'data/zerocap/forbidden_tokens.npy'
	
	cond_text: str = "Image of a"
	beam_size: int = 5
	target_seq_length: int = 15
	
	reset_context_delta: bool = True
	num_iterations: int = 5
	clip_loss_temperature: float = 0.01
	clip_scale: float = 1.0
	ce_scale: float = 0.2
	stepsize: float = 0.3
	grad_norm_factor: float = 0.9
	fusion_factor: float = 0.99
	
	repetition_penalty: float = 1.0
	end_token: str = "."
	end_factor: float = 1.01
	forbidden_factor: float = 20
	
	clip_vocab_size: int = 49408
	sot_token_id: int = 49406

class Config(ConfigBase, _Model):
	pass

class Cfg(Config):
	pass