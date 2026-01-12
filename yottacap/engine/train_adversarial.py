import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

def train_adversarial(
	dataloader,
	model: YottaCap,
	text_optimizer: Optimizer,
	img_optimizer: Optimizer,
	disc_optimizer: Optimizer,
):
	model.train()
	
	ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	
	tqdmloader = tqdm(dataloader, desc='Adversarial Training') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		text_emb: Tensor = batch['text_emb'].to(Cfg.device, non_blocking=True)
		image_emb: Tensor = batch['image_emb'].to(Cfg.device, non_blocking=True).float()
		
		features = model.extract_clip_features(text=text_emb)
		
		# Micro-step 1: Text Adapter & Decoder

		for __ in range(Cfg.micro_steps_iter[0]):
		
			S_text = model.get_text_latent(features['text_tokens']).to(Cfg.device, non_blocking=True)
			
			# 1. CE Loss
			logits = model.forward(S_text, text_emb[:, :-1])
			logits = logits[:, S_text.shape[1]:]
			loss_ce = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
			
			# 2. KL Loss
			loss_kl = S_text.pow(2).mean()
			
			# 3. Adversarial Loss (Fool D to think it's image)
			pred_fake = model.discriminator(S_text)
			loss_adv = F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
			
			loss_text = loss_ce + Cfg.kl_weight * loss_kl + Cfg.adv_weight * loss_adv
			
			text_optimizer.zero_grad()
			loss_text.backward()
			text_optimizer.step()
		
		# Micro-step 2: Image Adapter

		for __ in range(Cfg.micro_steps_iter[1]):

			S_img = model.get_image_latent(image_emb).to(Cfg.device, non_blocking=True)
			
			# SIM LOSS
			prefix = model.latent_proj(S_img)
			logits_img = model.gpt2.forward_logits(prefix)
			soft_embeds = model.gumbel_softmax(logits_img)
			
			T_text_recon = model.softemb_to_clip(soft_embeds).to(Cfg.device, non_blocking=True)
			T_text_recon = F.normalize(T_text_recon, dim=-1)
			
			T_image: Tensor = batch['image_feat'].to(Cfg.device, non_blocking=True).float()
			T_image = F.normalize(T_image, dim=-1)

			loss_sim = 1.0 - (T_text_recon * T_image).sum(dim=-1).mean()
			
			# KL LOSS
			loss_kl = S_img.pow(2).mean()
			
			loss_img = loss_sim + Cfg.kl_weight * loss_kl

			img_optimizer.zero_grad()
			loss_img.backward()
			img_optimizer.step()
		
		# Micro-step 3: Train Discriminator

		for __ in range(Cfg.micro_steps_iter[2]):
		
			S_img_det = S_img.detach()
			S_text_det = S_text.detach()
			
			pred_real = model.discriminator(S_img_det)
			pred_fake = model.discriminator(S_text_det)
			
			loss_d_real = F.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
			loss_d_fake = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
			loss_disc = (loss_d_real + loss_d_fake) * 0.5
			
			disc_optimizer.zero_grad()
			loss_disc.backward()
			disc_optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'LossText': loss_text.item(),
				'LossImg': loss_img.item(),
				'LossDisc': loss_disc.item(),
			})