import torch
from torch import Tensor
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from clipcap.config import Cfg
from clipcap.layer.clipcap import ClipCapModel

@torch.no_grad
def decode_batch(
	tokenizer: SimpleTokenizer,
	clipcap_model: ClipCapModel,
	prefix_embedding: Tensor,
) -> list[str]:
	clipcap_model.eval()
	batch_size = prefix_embedding.shape[0]
	emb_cat = prefix_embedding
	entry_length = Cfg.max_seq_length
	
	tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=prefix_embedding.device)
	
	for _ in range(entry_length):
		logits = clipcap_model.gpt.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :]
		
		next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
		
		tokens = torch.cat((tokens, next_token_id), dim=1)
		
		next_token_embed = clipcap_model.gpt.embed(next_token_id)
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
	clipcap_model: ClipCapModel,
	image_paths: list[Path],
) -> list[str]:
	clipcap_model.eval()
	device = next(clipcap_model.parameters()).device
	
	images = [preprocess(Image.open(p)) for p in image_paths]
	image_tensor = torch.stack(images).to(device)
	
	image_features = clip_model.encode_image(image_tensor).float()
	
	prefix_embedding = clipcap_model.clip_project(image_features)
	prefix_embedding = prefix_embedding.view(len(image_paths), Cfg.prefix_length, -1)
	
	return decode_batch(tokenizer, clipcap_model, prefix_embedding)