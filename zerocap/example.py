import clip
from clip.simple_tokenizer import SimpleTokenizer

from zerocap.config import Cfg
from zerocap.layer.gpt2 import GPT2
from zerocap.engine.decode import image_to_text
from utils.logger import logger

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=Cfg.device,
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('gpt2', 'loading'):
	gpt_model = GPT2().to(Cfg.device)
	gpt_model.eval()

for i in range(1, 9):
	image_path = Cfg.root/f'data/example/{i}.jpg'
	text = image_to_text(
		clip_model = clip_model,
		preprocess = preprocess,
		gpt_model = gpt_model,
		tokenizer = tokenizer,
		image_path = image_path
	)
	print(text)