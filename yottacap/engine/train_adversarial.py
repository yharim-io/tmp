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
	optimizer: Optimizer,
	disc_optimizer: Optimizer,
):
	model.train()
	
	ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	
	tqdmloader = tqdm(dataloader, desc='Adversarial Training') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		text_emb = batch['text_emb']
		image_emb = batch['image_emb']
		text_emb = text_emb.to(Cfg.device, non_blocking=True)
		image_emb = image_emb.to(Cfg.device, non_blocking=True)
		
		features = model.extract_clip_features(text=text_emb)
		
		# Micro-step 1: Text Adapter & Decoder
		
		S_text = model.get_text_latent(features['text_tokens'])
		
		# 1. CE Loss
		logits = model.forward(S_text, text_emb[:, :-1])
		logits = logits[:, S_text.shape[1]:]
		loss_ce = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
		
		# 2. Adversarial Loss (Fool D to think it's image)
		pred_fake = model.discriminator(S_text)
		loss_adv = F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
		
		# 3. Feature Constraint (prevent drift)
		S_text_pool = F.normalize(S_text.mean(dim=1), dim=-1)
		loss_clip_text = 1.0 - (S_text_pool * features['T_text']).sum(dim=-1).mean()
		
		loss_decoder = loss_ce + Cfg.adv_weight * loss_adv + Cfg.clip_loss_weight * loss_clip_text
		
		optimizer.zero_grad()
		loss_decoder.backward()
		optimizer.step()
		
		# Micro-step 2: Image Adapter

		S_img = model.get_image_latent(image_emb)
		S_img_pool = F.normalize(S_img.mean(dim=1), dim=-1)
		loss_sim = 1.0 - (S_img_pool * features['T_image']).sum(dim=-1).mean()

		optimizer.zero_grad()
		loss_sim.backward()
		optimizer.step()
		
		# Micro-step 3: Train Discriminator
		
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
				'LossCE': loss_ce.item(),
				'LossAdv': loss_adv.item(),
				'LossClipText': loss_clip_text.item(),
				'LossDecoder': loss_decoder.item(),
				'LossSim': loss_sim.item(),
				'LossDisc': loss_disc.item(),
			})