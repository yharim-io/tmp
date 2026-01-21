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

	sot_token = torch.tensor([[Cfg.sot_token_id]], device=text_concepts.device, dtype=torch.long)
	sot_emb = upcap_model.gpt2.embed(sot_token)

	# prefix_embeds = upcap_model.concepts_embed(text_concepts)
	# current_embeds = torch.cat([prefix_embeds, sot_emb], dim=1)
	
	global_embed, local_embed = upcap_model.concepts_embed(text_concepts)
	current_embeds = torch.cat([global_embed, sot_emb], dim=1)

	entry_length = Cfg.max_seq_length
	tokens = None
	past_key_values = None

	for _ in range(entry_length):
		
		# logits, past_key_values = upcap_model.gpt2.forward_embeds(
		# 	inputs_embeds=current_embeds,
		# 	past_key_values=past_key_values,
		# 	use_cache=True
		# )
		
		logits, past_key_values = upcap_model.gpt2.forward_embeds(
			inputs_embeds=current_embeds,
			encoder_hidden_states=local_embed,
			past_key_values=past_key_values,
			use_cache=True
		)

		logits: Tensor = logits[:, -1, :]

		next_token_id = torch.argmax(logits, -1).unsqueeze(0)
		
		next_token_embed = upcap_model.gpt2.embed(next_token_id)
		
		if tokens is None:
			tokens = next_token_id
		else:
			tokens = torch.cat((tokens, next_token_id), dim=1)
		
		if next_token_id.item() == Cfg.eos_token_id:
			break
		
		current_embeds = next_token_embed
	
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
	
	global_feat: Tensor = clip_model.encode_image(image_preprocessed).float()
	global_feat /= global_feat.norm(dim=-1, keepdim=True)
	
	concept_images = divider.process(image_path, bg=True, flatten=True)
	
	if concept_images.numel() > 0:
		concept_images = concept_images.permute(0, 3, 1, 2).float()
		concept_images = F.interpolate(concept_images, size=(224, 224), mode='bilinear', align_corners=False)
		concept_images /= 255.0
		mean = torch.tensor(Cfg.clip_mean, device=Cfg.device).view(1, 3, 1, 1)
		std = torch.tensor(Cfg.clip_std, device=Cfg.device).view(1, 3, 1, 1)
		concept_images = (concept_images - mean) / std
		
		local_feats = clip_model.encode_image(concept_images).float()
		local_feats /= local_feats.norm(dim=-1, keepdim=True)
		
		# sort by sim with global concept
		local_feats = local_feats[(local_feats @ global_feat.T).squeeze(-1).argsort(descending=True)]
		
		if local_feats.shape[0] > Cfg.max_concepts - 1:
			local_feats = local_feats[:Cfg.max_concepts - 1]
		
		text_concepts = torch.cat([global_feat.unsqueeze(1), local_feats.unsqueeze(0)], dim=1)
	else:
		text_concepts = global_feat.unsqueeze(1)
	
	pad_len = Cfg.max_concepts - text_concepts.shape[1]
	if pad_len > 0:
		zero_tokens = torch.zeros((1, 77), dtype=torch.long, device=Cfg.device)
		pad_feat = clip_model.encode_text(zero_tokens).float()
		pad_feat /= pad_feat.norm(dim=-1, keepdim=True)
		pad = pad_feat.expand(1, pad_len, -1)
		text_concepts = torch.cat([text_concepts, pad], dim=1)
		
	text = decode(tokenizer, upcap_model, text_concepts)
	return text