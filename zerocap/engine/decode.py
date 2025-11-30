import torch
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from zerocap.config import Cfg
from zerocap.layer.zerocap import ZeroCap

def image_to_text(
	clip_model: CLIP,
	preprocess: Compose,
	zerocap_model: ZeroCap,
	image_path: Path,
) -> str:
	
	zerocap_model.eval()
	
	with torch.no_grad():
		image = Image.open(image_path)
		image_preprocessed = preprocess(image).unsqueeze(0).to(Cfg.device)
		image_features = clip_model.encode_image(image_preprocessed).float()
		image_features = image_features / image_features.norm(dim=-1, keepdim=True)
	
	captions = zerocap_model.generate_text(
		image_features,
		Cfg.cond_text
	)
	
	with torch.no_grad():
		encoded_captions = [clip_model.encode_text(clip.tokenize(c).to(Cfg.device)).float() for c in captions]
		encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
		best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()
		best_caption = captions[best_clip_idx]
	
	return Cfg.cond_text + best_caption