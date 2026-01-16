import torch
from torchvision.transforms import Compose
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image
from pathlib import Path

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

@torch.no_grad
def image_to_text_batch(
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	model: YottaCap,
	image_paths: list[Path]
) -> list[str]:
	model.eval()
	
	images = [preprocess(Image.open(p)) for p in image_paths]
	image_tensor = torch.stack(images).to(Cfg.device)
	
	feats = model.extract_clip_features(image=image_tensor)
	S_img = model.image_adapter(feats['vit_tokens'])
	
	B = len(image_paths)
	generated = torch.zeros((B, 0), dtype=torch.long, device=Cfg.device)
	unfinished = torch.ones(B, dtype=torch.bool, device=Cfg.device)
	
	for _ in range(Cfg.max_seq_length):
		if generated.shape[1] == 0:
			sot = torch.tensor([[tokenizer.encoder['<|startoftext|>']]] * B, device=Cfg.device)
			emb_text = model.gpt2.embed(sot)
		else:
			sot = torch.tensor([[tokenizer.encoder['<|startoftext|>']]] * B, device=Cfg.device)
			full_seq = torch.cat([sot, generated], dim=1)
			emb_text = model.gpt2.embed(full_seq)
			
		inputs = torch.cat([S_img, emb_text], dim=1)
		logits = model.gpt2.forward_logits(inputs)
		next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
		
		generated = torch.cat([generated, next_tokens], dim=1)
		
		eos_id = tokenizer.encoder['<|endoftext|>']
		just_finished = (next_tokens.squeeze(1) == eos_id)
		unfinished = unfinished & (~just_finished)
		
		if (~unfinished).all():
			break
			
	results = []
	for i in range(B):
		tokens = generated[i].cpu().tolist()
		try:
			end_idx = tokens.index(eos_id)
			tokens = tokens[:end_idx]
		except ValueError:
			pass
		results.append(tokenizer.decode(tokens))
		
	return results