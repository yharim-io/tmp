import torch
from torch import Tensor
from torch.nn import functional as F
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image

from upcap.config import Cfg
from upcap.model.upcap import UpCap
from upcap.model.divider import Divider

@torch.no_grad
def decode(
	tokenizer: SimpleTokenizer,
	upcap_model: UpCap,
	text_concepts: Tensor
) -> str:
	
	upcap_model.eval()
	
	global_concept = text_concepts[:, :1]
	local_concepts = text_concepts[:, 1:]
	
	prefixes = upcap_model.attention(local_concepts, getattr(upcap_model, 'concepts_feat'))
	prefixes = torch.cat([global_concept, prefixes], dim=1)
	emb_cat = upcap_model.mlp(prefixes)
	
	entry_length = Cfg.max_seq_length
	tokens = None
	
	for _ in range(entry_length):
		
		logits = upcap_model.gpt2.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :]
		next_token_id = torch.argmax(logits, -1).unsqueeze(0)
		
		next_token_embed = upcap_model.gpt2.embed(next_token_id)
		
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
	upcap_model: UpCap,
	divider: Divider,
	image_path: Path,
) -> str:
	upcap_model.eval()
	
	image = Image.open(image_path)
	image_preprocessed = preprocess(image).unsqueeze(0).to(Cfg.device)
	
	global_feat = clip_model.encode_image(image_preprocessed).float()
	global_feat /= global_feat.norm(dim=-1, keepdim=True)
	
	concept_images = divider.process(image_path, bg=False)
	
	if concept_images.numel() > 0:
		concept_images = concept_images.permute(0, 3, 1, 2).float()
		concept_images = F.interpolate(concept_images, size=(224, 224), mode='bilinear', align_corners=False)
		concept_images /= 255.0
		mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=Cfg.device).view(1, 3, 1, 1)
		std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=Cfg.device).view(1, 3, 1, 1)
		concept_images = (concept_images - mean) / std
		
		local_feats = clip_model.encode_image(concept_images).float()
		local_feats /= local_feats.norm(dim=-1, keepdim=True)
		
		text_concepts = torch.cat([global_feat.unsqueeze(1), local_feats.unsqueeze(0)], dim=1)
	else:
		text_concepts = global_feat.unsqueeze(1)
		
	text = decode(tokenizer, upcap_model, text_concepts)
	return text