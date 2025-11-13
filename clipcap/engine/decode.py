import torch
from torch import Tensor
from torch.nn import functional as F
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from clip.simple_tokenizer import SimpleTokenizer
from pathlib import Path
from PIL import Image

from clipcap.config import Cfg
from clipcap.layer.clipcap import ClipCapModel

@torch.no_grad
def decode(
	tokenizer: SimpleTokenizer,
	clipcap_model: ClipCapModel,
	prefix_embedding: Tensor
) -> str:
	clipcap_model.eval()
	emb_cat = prefix_embedding
	entry_length = Cfg.max_seq_length
	tokens = None
	
	for _ in range(entry_length):
		
		logits = clipcap_model.gpt.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :]

		next_token_id = torch.argmax(logits, -1).unsqueeze(0)
		
		next_token_embed = clipcap_model.gpt.embed(next_token_id)
		
		if tokens is None:
			tokens = next_token_id
		else:
			tokens = torch.cat((tokens, next_token_id), dim=1)
		
		if next_token_id.item() == Cfg.eos_token_id:
			break
		
		emb_cat = torch.cat((emb_cat, next_token_embed), dim=1)
	
	try:
		output_list = list(tokens.squeeze().cpu().numpy())
		output = tokenizer.decode(output_list)
		output = output.replace('<|startoftext|>','').replace('<|endoftext|>','')
	except:
		output = 'error'
	
	return output

@torch.no_grad
def image_to_text(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	clipcap_model: ClipCapModel,
	image_path: Path,
) -> str:
	clipcap_model.eval()
	image = Image.open(image_path)
	image_preprocessed = preprocess(image).unsqueeze(0).to('cuda')
	image_feature: Tensor = clip_model.encode_image(image_preprocessed).float()
	prefix_embedding = clipcap_model.clip_project(image_feature)
	prefix_embedding = prefix_embedding.reshape(1, Cfg.prefix_length, -1)
	text = decode(tokenizer, clipcap_model, prefix_embedding)
	return text