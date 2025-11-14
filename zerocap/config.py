from utils.config import Config as BaseConfig

class _Decode:
	beam_size: int = 5
	top_k: int = 10
	clip_guidance_scale: float = 2.5
	max_seq_length: int = 30
	sot_token_id: int = 49406
	context_length: int = 77
	
class _Benchmark:
	batch_size: int = 64

class Config(BaseConfig, _Decode, _Benchmark):
	pass

class Cfg(Config):
	pass