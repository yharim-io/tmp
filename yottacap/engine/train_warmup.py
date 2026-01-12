import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm
from pathlib import Path

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

def train_warmup_step(
	dataloader,
	model: YottaCap,
	text_optimizer: Optimizer,
	image_optimizer: Optimizer,
	disc_optimizer: Optimizer,
	log_file: Path,
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
		
		S_text: Tensor = model.get_text_latent(features['text_tokens'])
		
		# CE Loss
		logits = model.forward(S_text, text_emb[:, :-1])
		logits = logits[:, S_text.shape[1]:]
		loss_ce = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
		
		# KL Loss
		loss_kl_text = (S_text.norm(dim=-1) - 1).pow(2).mean()
		
		loss_text = loss_ce + Cfg.kl_weight * loss_kl_text
		
		text_optimizer.zero_grad()
		loss_text.backward()
		text_optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'Loss': loss_text.item(),
				'LossCE': loss_ce.item(),
				'LossKL': loss_kl_text.item(),
			})

	set_freeze([model.text_adapter, model.discriminator], True)
	set_freeze([model.image_adapter], False)

	tqdmloader = tqdm(dataloader, desc='Image Alignment') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		image_emb = batch['image_emb']
		image_emb = image_emb.to(Cfg.device, non_blocking=True).float()
		
		S_img: Tensor = model.get_image_latent(image_emb).to(Cfg.device, non_blocking=True)

		# SIM Loss
		prefix = model.latent_proj(S_img)
		logits_img = model.gpt2.forward_logits(prefix)
		soft_embeds = model.gumbel_softmax(logits_img)
		
		T_text_recon: Tensor = model.softemb_to_clip(soft_embeds).to(Cfg.device, non_blocking=True)
		T_text_recon = F.normalize(T_text_recon, dim=-1)

		T_image: Tensor = batch['image_feat'].to(Cfg.device, non_blocking=True)
		T_image = F.normalize(T_image, dim=-1)
		
		loss_sim = 1.0 - (T_text_recon * T_image).sum(dim=-1).mean()
		
		# KL Loss
		loss_kl_img = (S_img.norm(dim=-1) - 1).pow(2).mean()
		
		loss_img = loss_sim + Cfg.kl_weight * loss_kl_img
		
		image_optimizer.zero_grad()
		loss_img.backward()
		image_optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'Loss': loss_img.item(),
				'LossSim': loss_sim.item(),
				'LossKL': loss_kl_img.item(),
			})
	
	set_freeze([model.text_adapter, model.image_adapter], True)
	set_freeze([model.discriminator], False)
	
	tqdmloader = tqdm(dataloader, desc='Discriminator Warmup') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		text_emb = batch['text_emb'].to(Cfg.device, non_blocking=True)
		image_emb = batch['image_emb'].to(Cfg.device, non_blocking=True).float()
		
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
		
		disc_optimizer.zero_grad()
		loss_disc.backward()
		disc_optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({'LossDisc': loss_disc.item()})
		
	set_freeze([model.text_adapter, model.image_adapter, model.discriminator], False)
	
	if Cfg.is_master:
		with open(log_file, 'a+') as f:
			f.writelines(f'\n\tLossText: {loss_text:.3f}, CE: {loss_ce:.3f}, TextKL: {loss_kl_text:.3f}')
			f.writelines(f'\n\tLossImage: {loss_img:.3f}, Sim: {loss_sim:.3f}, ImageKL: {loss_kl_img:.3f}')
			f.writelines(f'\n\tLossDisc: {loss_disc:.3f}\n')