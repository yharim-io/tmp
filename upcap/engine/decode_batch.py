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
	
	global_concept = text_concepts[:, :1]
	local_concepts = text_concepts[:, 1:]
	
	prefixes = upcap_model.attention(local_concepts, getattr(upcap_model, 'concepts_feat'))
	prefixes = torch.cat([global_concept, prefixes], dim=1)
	emb_cat = upcap_model.mlp(prefixes)
	
	entry_length = Cfg.max_seq_length
	tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=text_concepts.device)
	
	for _ in range(entry_length):
		
		logits = upcap_model.gpt2.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :]
		
		next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
		tokens = torch.cat((tokens, next_token_id), dim=1)
		
		next_token_embed = upcap_model.gpt2.embed(next_token_id)
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
	upcap_model: UpCap,
	divider: Divider,
	image_paths: list[Path]
) -> list[str]:
	
	upcap_model.eval()
	
	batch_concepts = []
	
	mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=Cfg.device).view(1, 3, 1, 1)
	std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=Cfg.device).view(1, 3, 1, 1)
	
	max_concepts = Cfg.max_concepts
	
	for path in image_paths:
		image = Image.open(path)
		image_pre = preprocess(image).unsqueeze(0).to(Cfg.device)
		
		global_feat = clip_model.encode_image(image_pre).float()
		global_feat /= global_feat.norm(dim=-1, keepdim=True)
		
		concept_images = divider.process(path, bg=False)
		
		if concept_images.numel() > 0:
			concept_images = concept_images.permute(0, 3, 1, 2).float()
			concept_images = F.interpolate(concept_images, size=(224, 224), mode='bilinear', align_corners=False)
			concept_images /= 255.0
			concept_images = (concept_images - mean) / std
			
			local_feats = clip_model.encode_image(concept_images).float()
			local_feats /= local_feats.norm(dim=-1, keepdim=True)
			
			if local_feats.shape[0] > max_concepts - 1:
				local_feats = local_feats[:max_concepts - 1]
			
			combined = torch.cat([global_feat, local_feats], dim=0)
		else:
			combined = global_feat
			
		batch_concepts.append(combined)
		
	padded_concepts = []
	for c in batch_concepts:
		pad_len = max_concepts - c.shape[0]
		if pad_len > 0:
			pad = torch.zeros((pad_len, c.shape[1]), device=c.device, dtype=c.dtype)
			c = torch.cat([c, pad], dim=0)
		padded_concepts.append(c)
		
	text_concepts = torch.stack(padded_concepts, dim=0)
	
	return decode_batch(tokenizer, upcap_model, text_concepts)