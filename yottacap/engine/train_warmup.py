import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

def set_freeze(modules: list[nn.Module], freeze: bool):
	req_grad = not freeze
	for mod in modules:
		for p in mod.parameters():
			p.requires_grad_(req_grad)

def train_warmup_step(
	dataloader,
	model: YottaCap,
	text_optimizer: Optimizer,
	image_optimizer: Optimizer,
	disc_optimizer: Optimizer,
	log_file: Path,
):
	model.train()
	model.clip_model.eval()
	
	ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	scaler = GradScaler('cuda')
	
	set_freeze([model.image_adapter, model.discriminator], True)
	set_freeze([model.text_adapter, model.gpt2], False)

	tqdmloader = tqdm(dataloader, desc='Text Priming') if Cfg.is_master else dataloader
	
	for batch in tqdmloader:
		text_emb: Tensor = batch['text_emb']
		text_emb = text_emb.to(Cfg.device, non_blocking=True)
		
		with autocast('cuda'):
			features = model.extract_clip_features(text=text_emb)
			
			S_text: Tensor = model.text_adapter(features['text_tokens'])
			
			# CE Loss
			logits = model.forward(S_text, text_emb[:, :-1])
			logits = logits[:, S_text.shape[1]:]
			loss_ce_text: Tensor = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
			
			loss_text = loss_ce_text
		
		text_optimizer.zero_grad()
		scaler.scale(loss_text).backward()
		scaler.step(text_optimizer)
		scaler.update()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'Loss': loss_text.item(),
				'LossCE': loss_ce_text.item(),
			})

	set_freeze([model.text_adapter, model.gpt2, model.discriminator], True)
	set_freeze([model.image_adapter], False)

	tqdmloader = tqdm(dataloader, desc='Image Alignment') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		image_emb = batch['image_emb']
		image_emb = image_emb.to(Cfg.device, non_blocking=True).float()
		
		with autocast('cuda'):
			S_img: Tensor = model.image_adapter(image_emb).to(Cfg.device, non_blocking=True)

			# SIM Loss
			# soft_embeds = model.gumbel_decode(S_img)
			
			# T_text_recon: Tensor = model.softemb_to_clip(soft_embeds).to(Cfg.device, non_blocking=True)
			# T_text_recon = F.normalize(T_text_recon, dim=-1)

			# T_image: Tensor = batch['image_feat'].to(Cfg.device, non_blocking=True)
			# T_image = F.normalize(T_image, dim=-1)
			
			# loss_sim = 1.0 - (T_text_recon * T_image).sum(dim=-1).mean()
			
			# loss_img = loss_sim
			
			# CE Loss
			logits = model.forward(S_img, text_emb[:, :-1])
			logits = logits[:, S_img.shape[1]:]
			loss_ce_img: Tensor = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
			
			loss_img = loss_ce_img
		
		image_optimizer.zero_grad()
		scaler.scale(loss_img).backward()
		scaler.step(image_optimizer)
		scaler.update()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'Loss': loss_img.item(),
				# 'LossSim': loss_sim.item(),
				'LossCE': loss_ce_img.item(),
			})
	
	set_freeze([model.text_adapter, model.gpt2, model.image_adapter], True)
	set_freeze([model.discriminator], False)
	
	tqdmloader = tqdm(dataloader, desc='Discriminator Warmup') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		text_emb = batch['text_emb'].to(Cfg.device, non_blocking=True)
		image_emb = batch['image_emb'].to(Cfg.device, non_blocking=True).float()
		
		with autocast('cuda'):
			with torch.no_grad():
				features = model.extract_clip_features(text=text_emb)
				S_text = model.text_adapter(features['text_tokens'])
				S_img = model.image_adapter(image_emb)
			
			S_text = S_text.detach()
			S_img = S_img.detach()
			
			pred_real_logits = model.discriminator(S_img)
			pred_fake_logits = model.discriminator(S_text)
			
			loss_real = F.binary_cross_entropy_with_logits(pred_real_logits, torch.ones_like(pred_real_logits))
			loss_fake = F.binary_cross_entropy_with_logits(pred_fake_logits, torch.zeros_like(pred_fake_logits))
			loss_disc = (loss_real + loss_fake) * 0.5
		
		disc_optimizer.zero_grad()
		scaler.scale(loss_disc).backward()
		scaler.step(disc_optimizer)
		scaler.update()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({'LossDisc': loss_disc.item()})
		
	set_freeze([model.text_adapter, model.image_adapter, model.discriminator], False)
	
	if Cfg.is_master:
		with open(log_file, 'a+') as f:
			f.writelines(f'\n\tLossText: {loss_text:.3f}, CE: {loss_ce_text:.3f}')
			# f.writelines(f'\n\tLossImage: {loss_img:.3f}, Sim: {loss_sim:.3f}')
			f.writelines(f'\n\tLossImage: {loss_img:.3f}, CE: {loss_ce_img:.3f}')
			f.writelines(f'\n\tLossDisc: {loss_disc:.3f}')
			f.writelines('\n')