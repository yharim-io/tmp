import torch
from torch import Tensor
from torch.nn import functional as F
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from zerocap.config import Cfg
from zerocap.layer.gpt2 import GPT2

@torch.no_grad()
def image_to_text_batch(
	clip_model: CLIP,
	preprocess: Compose,
	gpt_model: GPT2,
	tokenizer: SimpleTokenizer,
	image_paths: list[Path],
) -> list[str]:
	
	device = Cfg.device
	max_seq_len = Cfg.max_seq_length
	
	gpt_model.eval()
	clip_model.eval()
	
	images = [preprocess(Image.open(p)) for p in image_paths]
	image_tensor = torch.stack(images).to(device)
	image_features = clip_model.encode_image(image_tensor).float()
	
	batch_size = image_features.shape[0]
	tokens = torch.full((batch_size, 1), Cfg.eos_token_id, dtype=torch.long, device=device)
	
	for _ in range(max_seq_len):
		token_embeds = gpt_model.embed(tokens)
		logits = gpt_model.forward_embeds(token_embeds)
		logits = logits[:, -1, :]
		
		next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
		tokens = torch.cat((tokens, next_token_id), dim=1)
	
	output_texts = []
	token_lists = tokens.cpu().numpy().tolist()
	
	for seq in token_lists:
		try:
			seq = seq[1:]
			if Cfg.eos_token_id in seq:
				seq = seq[:seq.index(Cfg.eos_token_id)]
			text = tokenizer.decode(seq)
			text = text.replace('<|startoftext|>', '').split('<|endoftext|>')[0]
			output_texts.append(text)
		except Exception:
			output_texts.append("error")
			
	return output_texts