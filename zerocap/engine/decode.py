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

CLIP_VOCAB_SIZE = 49408

@torch.no_grad()
def get_top_k_logits(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
	probs = F.softmax(logits, dim=-1)
	top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
	return top_k_probs, top_k_indices

@torch.no_grad()
def image_to_text(
	clip_model: CLIP,
	preprocess: Compose,
	gpt_model: GPT2,
	tokenizer: SimpleTokenizer,
	image_path: Path,
) -> str:
	
	device = Cfg.device
	beam_size = Cfg.beam_size
	top_k = Cfg.top_k
	clip_guidance_scale = Cfg.clip_guidance_scale
	max_seq_len = Cfg.max_seq_length
	context_length = Cfg.context_length
	sot_token_id = Cfg.sot_token_id
	eos_token_id = Cfg.eos_token_id
	
	gpt_model.eval()
	clip_model.eval()
	
	image = Image.open(image_path)
	image_preprocessed = preprocess(image).unsqueeze(0).to(device)
	image_features = clip_model.encode_image(image_preprocessed).float()
	image_features /= image_features.norm(dim=-1, keepdim=True)

	beams = [{'tokens': [sot_token_id], 'gpt_score': 0.0, 'score': 0.0}]

	for _ in range(max_seq_len):
		new_beams = []
		for beam in beams:
			
			if beam['tokens'][-1] == eos_token_id and len(beam['tokens']) > 1:
				new_beams.append(beam)
				continue

			token_input = torch.tensor([beam['tokens']], device=device)
			token_embeds = gpt_model.embed(token_input)
			
			logits = gpt_model.forward_embeds(token_embeds)
			logits = logits[:, -1, :]
			
			# 屏蔽掉 CLIP 词汇表 (49408) 之外的 token
			logits[:, CLIP_VOCAB_SIZE:] = -float('inf')
			
			gpt_probs, gpt_indices = get_top_k_logits(logits, top_k)

			for i in range(top_k):
				next_token = gpt_indices[0, i].item()
				next_prob = gpt_probs[0, i].item()
				
				new_tokens = beam['tokens'] + [next_token]
				new_gpt_score = beam['gpt_score'] + torch.log(torch.tensor(next_prob)).item()
				
				padded_tokens = torch.full((1, context_length), 0, dtype=torch.long, device=device)
				current_len = len(new_tokens)
				
				if current_len > context_length:
					current_len = context_length
					new_tokens_for_clip = new_tokens[:context_length]
				else:
					new_tokens_for_clip = new_tokens
				
				padded_tokens[0, :current_len] = torch.tensor(new_tokens_for_clip, device=device)
				text_tokens = padded_tokens
				
				text_features = clip_model.encode_text(text_tokens)
				text_features /= text_features.norm(dim=-1, keepdim=True)
				
				clip_score = torch.sum(text_features * image_features, dim=-1).item()
				
				score = new_gpt_score + clip_guidance_scale * clip_score
				
				new_beams.append({'tokens': new_tokens, 'gpt_score': new_gpt_score, 'score': score})

		new_beams.sort(key=lambda x: x['score'], reverse=True)
		beams = new_beams[:beam_size]
		
		if all(b['tokens'][-1] == Cfg.eos_token_id for b in beams):
			break

	best_beam = beams[0]
	try:
		output_list = best_beam['tokens'][1:]
		if eos_token_id in output_list:
			output_list = output_list[:output_list.index(eos_token_id)]
		output = tokenizer.decode(output_list)
	except Exception:
		output = "error"
	
	return output