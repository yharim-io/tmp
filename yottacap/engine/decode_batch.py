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
def decode_batch(
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	clip_features: Tensor,
	entry_length: int = 30
) -> list[str]:
	yottacap_model.eval()
	batch_size = clip_features.shape[0]
	emb_cat = yottacap_model.mlp(clip_features).reshape(batch_size, 1, -1)
	tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=clip_features.device)
	
	for _ in range(entry_length):
		logits = yottacap_model.gpt2.forward_logits(inputs_embeds=emb_cat)
		next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
		tokens = torch.cat((tokens, next_token), dim=1)
		
		next_token_embed = yottacap_model.gpt2.embed(next_token)
		emb_cat = torch.cat((emb_cat, next_token_embed), dim=1)
	
	output_texts = []
	token_lists = tokens.cpu().numpy().tolist()
	
	for seq in token_lists:
		text = tokenizer.decode(seq)
		text = text.replace('<|startoftext|>', '').split('<|endoftext|>')[0]
		output_texts.append(text)
		
	return output_texts

@torch.no_grad
def image_to_text_batch(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	text_features: Tensor,
	image_paths: list[Path]
) -> list[str]:
	
	images = []
	for p in image_paths:
		images.append(preprocess(Image.open(p)))
	
	image_batch = torch.stack(images).to(Cfg.device)
	
	image_features = clip_model.encode_image(image_batch).float()
	image_features /= image_features.norm(dim=-1, keepdim=True)
	
	sim = image_features @ text_features.T.float()
	sim = (sim * 100).softmax(dim=-1)
	
	prefix_embedding = sim @ text_features.float()
	prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
	
	return decode_batch(tokenizer, yottacap_model, prefix_embedding)