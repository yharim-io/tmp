import torch
from torch import Tensor
from torch.utils.data import Dataset
import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from pathlib import Path
from PIL import Image
import tqdm

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

@torch.no_grad
def calc_text_features(
	clip_model: CLIP,
	dataset: Dataset,
	batch_size: int = 100
) -> Tensor:
	
	clip_model.eval()
	text_features = []
	
	for i in tqdm(range(len(dataset) // batch_size + 1), desc='Calculating text features'):
		start_idx = i * batch_size
		end_idx = min((i + 1) * batch_size, len(dataset))
		texts = [
			dataset[j]['text']
			for j in range(start_idx, end_idx)
		]
		token_ids = clip.tokenize(texts).to(Cfg.device)
		text_feature = clip_model.encode_text(token_ids)
		text_features.append(text_feature)
	
	text_features = torch.cat(text_features, dim=0)
	text_features /= text_features.norm(dim=-1, keepdim=True).float()
	
	return text_features

@torch.no_grad
def decode(
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	clip_features: Tensor
) -> str:
	
	yottacap_model.eval()
	emb_cat = yottacap_model.mlp(clip_features).reshape(1, 1, -1)
	tokens = None
	entry_length = Cfg.max_seq_length
	
	for _ in range(entry_length):
		logits = yottacap_model.gpt2.forward_logits(inputs_embeds=emb_cat)
		next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
		
		if tokens is None:
			tokens = next_token_id
		else:
			tokens = torch.cat((tokens, next_token_id), dim=1)
		
		if next_token_id.item() == Cfg.eos_token_id:
			break
		
		next_token_embed = yottacap_model.gpt2.embed(next_token_id)
		emb_cat = torch.cat((emb_cat, next_token_embed), dim=1)
		
	try:
		output_list = list(tokens.squeeze().cpu().numpy())
		output = tokenizer.decode(output_list)
		output = output.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
	except:
		output = 'error'
		
	return output

@torch.no_grad
def image_to_text(
	clip_model: CLIP,
	preprocess: Compose,
	tokenizer: SimpleTokenizer,
	yottacap_model: YottaCap,
	text_features: Tensor,
	image_path: Path
) -> str:
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to(Cfg.device)
	
	image_feature: Tensor = clip_model.encode_image(image).float()
	image_feature /= image_feature.norm(dim=-1, keepdim=True)
	
	sim = image_feature @ text_features.T.float()
	sim = (sim * 100).softmax(dim=-1)
	prefix_embedding = sim @ text_features.float()
	prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
	
	text = decode(tokenizer, yottacap_model, image_feature)
	return text