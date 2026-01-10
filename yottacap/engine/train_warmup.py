import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

def train_warmup(
	dataloader,
	model: YottaCap,
	optimizer: Optimizer,
):
	model.train()
	ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	
	def set_freeze(modules: list[nn.Module], freeze: bool):
		for mod in modules:
			for p in mod.parameters():
				p.requires_grad = not freeze
	
	if Cfg.is_master:
		print('Text Priming...')
	set_freeze([model.image_adapter, model.discriminator], True)
	set_freeze([model.text_adapter], False)
	
	for i, batch in enumerate(dataloader):
		text_input: Tensor = batch.get('text_emb', None)
		if text_input is None:
			continue
		text_input = text_input.to(Cfg.device)
		
		features = model.extract_clip_features(text=text_input)
		if 'text_tokens' not in features:
			continue
			
		S_text = model.get_text_latent(features['text_tokens'])
		logits = model.forward(S_text, text_input[:, :-1])
		loss = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_input[:, 1:].flatten())
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if Cfg.is_master:
			print(f'    Step {i} Text Loss: {loss.item():.4f}')
	
	if Cfg.is_master:
		print('Image Alignment...')
	set_freeze([model.text_adapter, model.discriminator], True)
	set_freeze([model.image_adapter], False)
	
	for i, batch in enumerate(dataloader):
		image_input = batch.get('image', None)
		if image_input is not None:
			continue
		image_input = image_input.to(Cfg.device)
		
		features = model.get_image_latent(features['vit_tokens'])
		if 'vit_tokens' not in features:
			continue
		
		S_img = model.extract_clip_features(image=image_input)
		S_img_pool = F.normalize(S_img.mean(dim=1), dim=-1)
		T_image = features['T_image']
		
		loss = 1.0 - (S_img_pool * T_image).sum(dim=-1).mean()
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if Cfg.is_master:
			print(f'    Step {i} Image Loss: {loss.item():.4f}')
	
	if Cfg.is_master:
		print('Discriminator Init...')
	set_freeze([model.text_adapter, model.image_adapter], True)
	set_freeze([model.discriminator], False)
	
	for i, batch in enumerate(dataloader):
		text_input = batch.get('text_emb', None)
		image_input = batch.get('image', None)
		
		if text_input is None or image_input is None:
			continue
		
		text_input = text_input.to(Cfg.device)
		image_input = image_input.to(Cfg.device)
		
		with torch.no_grad():
			features = model.extract_clip_features(image=image_input, text=text_input)
			S_text = model.get_text_latent(features['text_tokens'])
			S_img = model.get_image_latent(features['vit_tokens'])
		
		S_text = S_text.detach()
		S_img = S_img.detach()
		
		pred_real = model.discriminator(S_img)
		pred_fake = model.discriminator(S_text)
		
		loss_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
		loss_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
		loss = (loss_real + loss_fake) * 0.5
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if Cfg.is_master:
			print(f'    Step {i} Disc Loss: {loss.item():.4f}')
	
	set_freeze([model.text_adapter, model.image_adapter, model.discriminator], False)