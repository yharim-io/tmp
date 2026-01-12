import torch
from torch import Tensor
from torchvision.transforms import Compose
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image
from pathlib import Path

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

@torch.no_grad
def image_to_text(
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	model: YottaCap,
	image_path: Path
) -> str:
	
	model.eval()
	
	image = Image.open(image_path)
	image_tensor = preprocess(image).unsqueeze(0).to(Cfg.device)
	
	feats = model.extract_clip_features(image=image_tensor)
	S_img = model.get_image_latent(feats['vit_tokens'])
	
	prefix = model.latent_proj(S_img)
	
	generated = torch.tensor([], dtype=torch.long, device=Cfg.device).unsqueeze(0)
	curr_input = torch.tensor([[tokenizer.encoder['<|startoftext|>']]], dtype=torch.long, device=Cfg.device)
	
	for _ in range(Cfg.max_seq_length):
		emb_prefix = prefix
		emb_text = model.gpt2.embed(curr_input)
		if generated.shape[1] > 0:
			emb_gen = model.gpt2.embed(generated)
			emb_text = torch.cat([emb_text, emb_gen], dim=1)
			
		inputs = torch.cat([emb_prefix, emb_text], dim=1)
		logits = model.gpt2.forward_logits(inputs)
		next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
		
		if next_token.item() == tokenizer.encoder['<|endoftext|>']:
			break
			
		generated = torch.cat([generated, next_token], dim=1)
		
	tokens = generated.squeeze().cpu().tolist()
	return tokenizer.decode(tokens)