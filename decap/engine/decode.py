import torch
from torch import Tensor
from torch.nn import functional as F
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from dataset import CocoDataset
from decap.layer.decap import DeCap
from decap.config import Config

@torch.no_grad
def get_text_features(
	clip_model: CLIP,
	dataset: CocoDataset,
	batch_size: int = 100
) -> Tensor:
	
	clip_model.eval()
	text_features = []
	
	for i in tqdm(range(len(dataset) // batch_size + 1)):
		texts = dataset.captions[i * batch_size : (i + 1) * batch_size]
		token_ids = clip.tokenize(texts).to('cuda')
		text_feature = clip_model.encode_text(token_ids)
		text_features.append(text_feature)
	
	text_features = torch.cat(text_features, dim=0)
	text_features /= text_features.norm(dim=-1, keepdim=True).float()
	
	return text_features

@torch.no_grad
def decode(
	tokenizer: SimpleTokenizer,
	decap_model: DeCap,
	clip_features: Tensor
) -> str:
	
	decap_model.eval()
	emb_cat = decap_model.mlp(clip_features).reshape(1, 1, -1)
	entry_length = 30
	temperature = 1.0
	tokens = None
	
	for _ in range(entry_length):
		logits = decap_model.gpt2.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :] / temperature
		logits = F.softmax(logits, dim=1)
		next_token_id = torch.argmax(logits, -1).unsqueeze(0)
		next_token_embed = decap_model.gpt2.embed(next_token_id)
		
		if tokens is None:
			tokens = next_token_id
		else:
			tokens = torch.cat((tokens, next_token_id), dim=1)
		
		if next_token_id.item() == Config.model.eos_token_id:
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
	decap_model: DeCap,
	text_features: Tensor,
	image_path: Path
) -> str:
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to('cuda')
	image_feature: Tensor = clip_model.encode_image(image).float()
	image_feature /= image_feature.norm(dim=-1, keepdim=True)
	sim = image_feature @ text_features.T.float()
	sim = (sim * 100).softmax(dim=-1)
	prefix_embedding = sim @ text_features.float()
	prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
	text = decode(tokenizer, decap_model, prefix_embedding)
	return text
