from clip.model import CLIP
from torchvision.transforms import Compose
from pathlib import Path

from zerocap.layer.zerocap import ZeroCap
from zerocap.engine.decode import image_to_text

def image_to_text_batch(
	clip_model: CLIP,
	preprocess: Compose,
	zerocap_model: ZeroCap,
	image_paths: list[Path],
) -> list[str]:
	
	zerocap_model.eval()
	clip_model.eval()
	
	output_texts = []
	
	for image_path in image_paths:
		try:
			text = image_to_text(
				clip_model=clip_model,
				preprocess=preprocess,
				zerocap_model=zerocap_model,
				image_path=image_path
			)
			output_texts.append(text)
		except Exception:
			output_texts.append("error")
			
	return output_texts