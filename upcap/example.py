import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

from upcap.config import Cfg
from upcap.model.upcap import UpCap
from upcap.model.divider import Divider
from upcap.engine.decode import image_to_text
from upcap.engine.decode_batch import image_to_text_batch
from utils.logger import logger

with logger('clip', 'loading'):
	clip_model, preprocess = clip.load(
		name=Cfg.clip_pretrained_path,
		device=Cfg.device,
		jit=False
	)
	clip_model.eval()
	tokenizer = SimpleTokenizer()

with logger('divider', 'loading'):
	divider = Divider()

with logger('upcap', 'loading'):
	upcap_model = UpCap(
		enable_concepts_global_buffer=False,
		enable_concepts_local_buffer=True,
	)
	static_dict = torch.load(
		Cfg.root/'data/upcap/coco/001.pt',
		map_location='cpu',
		weights_only=True
	)
	if any(k.startswith('_orig_mod.') for k in static_dict.keys()):
		static_dict = {
			(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
			for k, v in static_dict.items()
		}
	# static_dict.pop('concepts_feat', None) # concepts_feat deprecated
	upcap_model.load_state_dict(static_dict)
	upcap_model = upcap_model.to(Cfg.device)
	upcap_model.eval()

# def sequential_test():
# 	for i in range(1, 10):
# 		text = image_to_text(
# 			clip_model = clip_model,
# 			preprocess = preprocess,
# 			tokenizer = tokenizer,
# 			upcap_model = upcap_model,
# 			divider = divider,
# 			image_path = Cfg.root/f'data/example/{i}.jpg'
# 		)
# 		print(text)

def parallel_test():
	texts = image_to_text_batch(
		clip_model = clip_model,
		preprocess = preprocess,
		tokenizer = tokenizer,
		upcap_model = upcap_model,
		divider = divider,
		image_paths = [
			Cfg.root/f'data/example/{i}.jpg'
			for i in range(1, 10)
		],
		global_attn=False,
		local_attn=False,
	)
	for t in texts:
		print(t)

if __name__ == '__main__':
	# sequential_test()
	parallel_test()