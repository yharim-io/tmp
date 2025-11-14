import torch
from torch import nn
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer

from zerocap.layer.gpt2 import GPT2
from zerocap.config import Cfg

class ZeroCapGenerator:

	def __init__(
		self,
		clip_model: CLIP,
		gpt_model: GPT2,
		tokenizer: SimpleTokenizer,
	):
		self.clip_model = clip_model
		self.gpt_model = gpt_model
		self.tokenizer = tokenizer
		self.device = Cfg.device
		
		self.gpt_model.eval()
		self.clip_model.eval()