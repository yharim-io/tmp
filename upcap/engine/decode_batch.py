import torch
from torch import Tensor
from torch.nn import functional as F
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from upcap.config import Cfg
from upcap.model.upcap import UpCap
from upcap.model.divider import Divider

@torch.no_grad
def decode_batch(
	tokenizer: SimpleTokenizer,
	upcap_model: UpCap,
	text_concepts: Tensor
) -> list[str]:
	
	upcap_model.eval()
	batch_size = text_concepts.shape[0]
	
	# global_concept = text_concepts[:, :1]
	# local_concepts = text_concepts[:, 1:]
	# prefixes = upcap_model.attention(local_concepts, upcap_model.concepts_feat)
	# prefixes = torch.cat([global_concept, prefixes], dim=1)
	
	prefixes = upcap_model.attention(text_concepts, upcap_model.concepts_feat)
	
	prefix_embeds = upcap_model.mlp(prefixes)
	
	sot_token = torch.full((batch_size, 1), Cfg.sot_token_id, device=text_concepts.device, dtype=torch.long)
	sot_emb = upcap_model.gpt2.embed(sot_token)
	
	current_embeds = torch.cat([prefix_embeds, sot_emb], dim=1)
	
	entry_length = Cfg.max_seq_length
	tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=text_concepts.device)
	
	past_key_values = None

	for _ in range(entry_length):
		
		logits, past_key_values = upcap_model.gpt2.forward_embeds(inputs_embeds=current_embeds, past_key_values=past_key_values, use_cache=True)
		logits = logits[:, -1, :]
		
		next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
		tokens = torch.cat((tokens, next_token_id), dim=1)
		
		current_embeds = upcap_model.gpt2.embed(next_token_id)
	
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
	upcap_model: UpCap,
	divider: Divider,
	image_paths: list[Path]
) -> list[str]:
	
	upcap_model.eval()
	
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=Cfg.device).view(1, 3, 1, 1)
	std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=Cfg.device).view(1, 3, 1, 1)
	max_concepts = Cfg.max_concepts
	
	# 1. Global Features
	images = [preprocess(Image.open(p)) for p in image_paths]
	image_tensor = torch.stack(images).to(Cfg.device)
	global_feats = clip_model.encode_image(image_tensor).float()
	global_feats /= global_feats.norm(dim=-1, keepdim=True)
	
	# 2. Local Features (Batch Process)
	concepts_list = divider.process_batch(image_paths, bg=False, flatten=False)
	
	all_concepts = []
	counts = []
	for c in concepts_list:
		if c.numel() > 0:
			all_concepts.append(c)
			counts.append(c.shape[0])
		else:
			counts.append(0)
	
	if all_concepts:
		batch_concepts = torch.cat(all_concepts, dim=0)
		batch_concepts = batch_concepts.permute(0, 3, 1, 2).float()
		batch_concepts = F.interpolate(batch_concepts, size=(224, 224), mode='bilinear', align_corners=False)
		batch_concepts /= 255.0
		batch_concepts = (batch_concepts - mean) / std
		
		all_local_feats = clip_model.encode_image(batch_concepts).float()
		all_local_feats /= all_local_feats.norm(dim=-1, keepdim=True)
		
		local_feats_list = torch.split(all_local_feats, counts)
	else:
		local_feats_list = [torch.empty(0, device=Cfg.device) for _ in counts]

	# 3. Combine and Pad
	
	zero_tokens = torch.zeros((1, 77), dtype=torch.long, device=Cfg.device)
	pad_feat = clip_model.encode_text(zero_tokens).float()
	pad_feat /= pad_feat.norm(dim=-1, keepdim=True)
	
	padded_concepts = []
	
	for g_feat, l_feats in zip(global_feats, local_feats_list):
		if l_feats.numel() > 0:
			# sort by sim with global concept
			l_feats = l_feats[(l_feats @ g_feat).argsort(descending=True)]
			if l_feats.shape[0] > max_concepts - 1:
				l_feats = l_feats[:max_concepts - 1]
			combined = torch.cat([g_feat.unsqueeze(0), l_feats], dim=0)
		else:
			combined = g_feat.unsqueeze(0)
		
		pad_len = max_concepts - combined.shape[0]
		if pad_len > 0:
			# [Fix] Use pad_feat instead of zeros
			pad = pad_feat.expand(pad_len, -1)
			combined = torch.cat([combined, pad], dim=0)
		padded_concepts.append(combined)

	text_concepts = torch.stack(padded_concepts, dim=0)
	
	return decode_batch(tokenizer, upcap_model, text_concepts)