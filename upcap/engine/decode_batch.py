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

def _autocast_ctx():
	if Cfg.device.type != 'cuda':
		return torch.autocast(device_type='cpu', enabled=False)
	return torch.autocast(device_type='cuda', dtype=torch.float16)

def _get_pad_feat(clip_model: CLIP) -> Tensor:
	pad_feat = getattr(clip_model, '_upcap_pad_feat', None)
	if pad_feat is None or pad_feat.device != Cfg.device:
		zero_tokens = torch.zeros((1, 77), dtype=torch.long, device=Cfg.device)
		pad_feat = clip_model.encode_text(zero_tokens).float()
		pad_feat /= pad_feat.norm(dim=-1, keepdim=True)
		setattr(clip_model, '_upcap_pad_feat', pad_feat)
	return pad_feat

@torch.inference_mode()
def decode_batch(
	tokenizer: SimpleTokenizer,
	upcap_model: UpCap,
	text_concepts: Tensor,
	global_attn: bool = False,
	local_attn: bool = False,
) -> list[str]:
	
	upcap_model.eval()
	batch_size = text_concepts.shape[0]
	
	global_feat = text_concepts[:, :1]
	local_feat = text_concepts[:, 1:]

	global_emb, local_emb = upcap_model.project_features(global_feat, local_feat, global_attn, local_attn)
	
	sot_token = torch.full((batch_size, 1), Cfg.sot_token_id, device=text_concepts.device, dtype=torch.long)
	sot_emb = upcap_model.embed_tokens(sot_token)

	inputs_embeds, cross_states = upcap_model.assemble_structure(global_emb, local_emb, sot_emb)
	
	entry_length = Cfg.max_seq_length
	tokens = torch.full((batch_size, entry_length), Cfg.eos_token_id, dtype=torch.long, device=text_concepts.device)
	past_key_values = None

	with _autocast_ctx():
		for t in range(entry_length):
			
			logits, past_key_values = upcap_model.forward(inputs_embeds, cross_states, past_key_values)

			logits = logits[:, -1, :]

			next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
			tokens[:, t] = next_token_id.squeeze(1)
			if bool(torch.all(next_token_id.squeeze(1) == Cfg.eos_token_id)):
				break
			
			inputs_embeds = upcap_model.embed_tokens(next_token_id)

	output_texts = []
	token_lists = tokens.cpu().numpy().tolist()
	
	for seq in token_lists:
		text = tokenizer.decode(seq)
		text = text.replace('<|startoftext|>', '').split('<|endoftext|>')[0]
		output_texts.append(text)
	
	for i in range(len(output_texts)):
		if len(output_texts[i]) >= 2 and output_texts[i][-2] not in {'.', '!', '?'}:
			output_texts[i] += '. '

	return output_texts

@torch.inference_mode()
def image_tensor_to_text_batch(
	clip_model: CLIP,
	tokenizer: SimpleTokenizer,
	upcap_model: UpCap,
	divider: Divider,
	image_tensor: Tensor,
	image_paths: list[Path],
	image_rgbs: list | None = None,
	global_attn: bool = False,
	local_attn: bool = False,
) -> list[str]:
	
	upcap_model.eval()
	clip_model.eval()
	
	mean = torch.tensor(Cfg.clip_mean, device=Cfg.device).view(1, 3, 1, 1)
	std = torch.tensor(Cfg.clip_std, device=Cfg.device).view(1, 3, 1, 1)
	
	with _autocast_ctx():
		# 1. Global Features
		image_tensor = image_tensor.to(Cfg.device, non_blocking=True)
		global_feats = clip_model.encode_image(image_tensor).float()
		global_feats /= global_feats.norm(dim=-1, keepdim=True)
		
		# 2. Local Features (Batch Process)
		concepts_list = divider.process_batch(
			image_paths=image_paths,
			image_rgbs=image_rgbs,
			bg=True,
			output_size=224,
			flatten=False
		)
		
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
			batch_concepts = batch_concepts.to(Cfg.device, non_blocking=True)
			
			all_local_feats = clip_model.encode_image(batch_concepts).float()
			all_local_feats /= all_local_feats.norm(dim=-1, keepdim=True)
			
			local_feats_list = torch.split(all_local_feats, counts)
		else:
			local_feats_list = [torch.empty(0, device=Cfg.device) for _ in counts]

		# 3. Combine and Pad

		max_concepts = Cfg.max_concepts
		pad_feat = _get_pad_feat(clip_model)
		
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
	
	return decode_batch(
		tokenizer,
		upcap_model,
		text_concepts,
		global_attn=global_attn,
		local_attn=local_attn,
	)

@torch.inference_mode()
def image_to_text_batch(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	upcap_model: UpCap,
	divider: Divider,
	image_paths: list[Path],
	global_attn: bool = False,
	local_attn: bool = False,
) -> list[str]:
	images = [preprocess(Image.open(p).convert('RGB')) for p in image_paths]
	image_tensor = torch.stack(images)
	return image_tensor_to_text_batch(
		clip_model=clip_model,
		tokenizer=tokenizer,
		upcap_model=upcap_model,
		divider=divider,
		image_tensor=image_tensor,
		image_paths=image_paths,
		global_attn=global_attn,
		local_attn=local_attn,
	)