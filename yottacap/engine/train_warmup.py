import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm

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
	
	set_freeze([model.image_adapter, model.discriminator], True)
	set_freeze([model.text_adapter], False)

	tqdmloader = tqdm(dataloader, desc='Text Priming') if Cfg.is_master else dataloader
	
	for batch in tqdmloader:
		text_emb: Tensor = batch['text_emb']
		text_emb = text_emb.to(Cfg.device, non_blocking=True)
		
		features = model.extract_clip_features(text=text_emb)
		if 'text_tokens' not in features:
			continue
			
		S_text = model.get_text_latent(features['text_tokens'])
		logits = model.forward(S_text, text_emb[:, :-1])
		logits = logits[:, S_text.shape[1]:]
		loss_ce = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
		
		optimizer.zero_grad()
		loss_ce.backward()
		optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({'LossCE': loss_ce.item()})

	set_freeze([model.text_adapter, model.discriminator], True)
	set_freeze([model.image_adapter], False)
	
	tqdmloader = tqdm(dataloader, desc='Image Alignment') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		image_emb = batch['image_emb']
		image_emb = image_emb.to(Cfg.device, non_blocking=True)
		
		S_img = model.get_image_latent(image_emb)
		S_img_pool = F.normalize(S_img.mean(dim=1), dim=-1)
		T_image = batch['image_feat']
		
		loss_sim = 1.0 - (S_img_pool * T_image).sum(dim=-1).mean()
		
		optimizer.zero_grad()
		loss_sim.backward()
		optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({'LossSim': loss_sim.item()})
	
	set_freeze([model.text_adapter, model.image_adapter], True)
	set_freeze([model.discriminator], False)
	
	tqdmloader = tqdm(dataloader, desc='Discriminator Warmup') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		text_emb = batch['text_emb']
		image_emb = batch['image_emb']
		text_emb = text_emb.to(Cfg.device, non_blocking=True)
		image_emb = image_emb.to(Cfg.device, non_blocking=True)
		
		with torch.no_grad():
			features = model.extract_clip_features(text=text_emb)
			S_text = model.get_text_latent(features['text_tokens'])
			S_img = model.get_image_latent(image_emb)
		
		S_text = S_text.detach()
		S_img = S_img.detach()
		
		pred_real = model.discriminator(S_img)
		pred_fake = model.discriminator(S_text)
		
		loss_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
		loss_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
		loss_disc = (loss_real + loss_fake) * 0.5
		
		optimizer.zero_grad()
		loss_disc.backward()
		optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({'LossDisc': loss_disc.item()})
		
	set_freeze([model.text_adapter, model.image_adapter, model.discriminator], False)