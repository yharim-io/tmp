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
		text_emb = batch['text_emb']
		image_emb = batch['image_emb']
		T_image = batch['image_feat']
		text_emb = text_emb.to(Cfg.device, non_blocking=True)
		image_emb = image_emb.to(Cfg.device, non_blocking=True)
		T_image = T_image.to(Cfg.device, non_blocking=True)
		
		features = model.extract_clip_features(text=text_emb)
		
		with 'Micro-step 1': # Micro-step 1: Text Adapter & Decoder
		
			S_text = model.get_text_latent(features['text_tokens'])
			
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
		
		with 'Micro-step 2': # Micro-step 2: Image Adapter

			S_img = model.get_image_latent(image_emb)
			
			prefix = model.latent_proj(S_img)
			logits_img = model.gpt2.forward_logits(prefix)
			soft_embeds = model.gumbel_softmax(logits_img)
			
			T_text_recon = model.project_to_clip(soft_embeds)
			T_text_recon = F.normalize(T_text_recon, dim=-1)
			
			loss_sim = 1.0 - (T_text_recon * T_image).sum(dim=-1).mean()
			
			loss_kl = S_img.pow(2).mean()
			
			loss_img = loss_sim + Cfg.kl_weight * loss_kl

			img_optimizer.zero_grad()
			loss_img.backward()
			img_optimizer.step()
		
		with 'Micro-step 3': # Micro-step 3: Train Discriminator
		
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