import torch
from torch import Tensor
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

@torch.no_grad
def decode(
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	clip_features: Tensor
) -> str:
	
	yottacap_model.eval()
	emb_cat = yottacap_model.mlp(clip_features).reshape(1, 1, -1)
	tokens = None
	entry_length = Cfg.max_seq_length
	
	for _ in range(entry_length):
		logits = yottacap_model.gpt2.forward_embeds(inputs_embeds=emb_cat)
		next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
		
		if tokens is None:
			tokens = next_token_id
		else:
			tokens = torch.cat((tokens, next_token_id), dim=1)
		
		if next_token_id.item() == Cfg.eos_token_id:
			break
		
		next_token_embed = yottacap_model.gpt2.embed(next_token_id)
		emb_cat = torch.cat((emb_cat, next_token_embed), dim=1)
		
	try:
		output_list = list(tokens.squeeze().cpu().numpy())
		output = tokenizer.decode(output_list)
		output = output.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
	except:
		output = 'error'
		
	return output

@torch.no_grad
def image_to_text(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	image_path: Path
) -> str:
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to(Cfg.device)
	
	image_feature: Tensor = clip_model.encode_image(image).float()
	image_feature /= image_feature.norm(dim=-1, keepdim=True)
	
	text = decode(tokenizer, yottacap_model, image_feature)
	return text