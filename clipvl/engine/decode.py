import torch
from torch import Tensor
from torch.nn import functional as F
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from clip.simple_tokenizer import SimpleTokenizer
from pathlib import Path
from PIL import Image

from clipvl.config import Cfg
from clipvl.layer.clipvl import ClipVLModel

@torch.no_grad
def decode(
	tokenizer: SimpleTokenizer,
	clipvl_model: ClipVLModel,
	prefix_embedding: Tensor
) -> str:

	clipvl_model.eval()
	emb_cat = prefix_embedding
	entry_length = Cfg.max_seq_length
	temperature = 1.0
	tokens = None
	
	for _ in range(entry_length):
		
		logits = clipvl_model.gpt.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :] / temperature
		logits = F.softmax(logits, dim=1)
		next_token_id = torch.argmax(logits, -1).unsqueeze(0)
		
		next_token_embed = clipvl_model.gpt.embed(next_token_id)
		
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
	clipvl_model: ClipVLModel,
	image_path: Path,
) -> str:
	clipvl_model.eval()
	image = Image.open(image_path)
	image_preprocessed = preprocess(image).unsqueeze(0).to('cuda')
	
	captured = { 'value': None }
	def hook_fn(module, input, output):
		captured['value'] = output
	
	handle = clip_model.visual.transformer.register_forward_hook(hook_fn)
	
	clip_model.encode_image(image_preprocessed)
	
	handle.remove()
	
	image_emb = captured['value'].permute(1, 0, 2)
	image_emb = clip_model.visual.ln_post(image_emb).float()

	prefix_embedding = clipvl_model.clip_project(image_emb)
	
	text = decode(tokenizer, clipvl_model, prefix_embedding)
	return text
