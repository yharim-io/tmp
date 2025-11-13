import torch
from torch import Tensor
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from clipvl.config import Cfg
from clipvl.layer.clipvl import ClipVLModel

@torch.no_grad
def decode_batch(
	tokenizer: SimpleTokenizer,
	clipvl_model: ClipVLModel,
	prefix_embedding: Tensor,
) -> list[str]:
	clipvl_model.eval()
	batch_size = prefix_embedding.shape[0]
	emb_cat = prefix_embedding
	entry_length = Cfg.max_seq_length
	
	tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=prefix_embedding.device)
	
	for _ in range(entry_length):
		logits = clipvl_model.gpt.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :]
		
		next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
		
		tokens = torch.cat((tokens, next_token_id), dim=1)
		
		next_token_embed = clipvl_model.gpt.embed(next_token_id)
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
	clipvl_model: ClipVLModel,
	image_paths: list[Path],
) -> list[str]:
	clipvl_model.eval()
	device = next(clipvl_model.parameters()).device
	
	images = [preprocess(Image.open(p)) for p in image_paths]
	image_tensor = torch.stack(images).to(device)
	
	captured = {'value': None}
	def hook_fn(module, input, output):
		captured['value'] = output
	
	handle = clip_model.visual.transformer.register_forward_hook(hook_fn)
	
	clip_model.encode_image(image_tensor)
	
	handle.remove()
	
	image_emb = captured['value'].permute(1, 0, 2)
	image_emb = clip_model.visual.ln_post(image_emb).float()
	
	prefix_embedding = clipvl_model.clip_project(image_emb)
	
	return decode_batch(tokenizer, clipvl_model, prefix_embedding)