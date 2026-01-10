import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer

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
	
	for i, batch in enumerate(dataloader):
		text_input = batch.get('text_emb', None).to(Cfg.device)
		image_input = batch.get('image', None).to(Cfg.device)
		
		features = model.extract_clip_features(image=image_input, text=text_input)
		
		# --- Phase 1: Train Generator (Adapters) ---
		loss_gen = torch.tensor(0.0, device=Cfg.device)
		
		S_text = model.get_text_latent(features['text_tokens'])
		
		# 1. CE Loss
		logits = model.forward(S_text, text_input[:, :-1])
		loss_ce = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_input[:, 1:].flatten())
		
		# 2. Adversarial Loss (Fool D to think it's image)
		pred_fake = model.discriminator(S_text)
		loss_adv = F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
		
		# 3. Feature Constraint (prevent drift)
		S_text_pool = F.normalize(S_text.mean(dim=1), dim=-1)
		loss_clip_text = 1.0 - (S_text_pool * features['T_text']).sum(dim=-1).mean()
		
		loss_gen += loss_ce + Cfg.adv_weight * loss_adv + Cfg.clip_loss_weight * loss_clip_text
		
		
		S_img = model.get_image_latent(features['vit_tokens'])
		S_img_pool = F.normalize(S_img.mean(dim=1), dim=-1)
		loss_align = 1.0 - (S_img_pool * features['T_image']).sum(dim=-1).mean()
		
		loss_gen += Cfg.clip_loss_weight * loss_align

		optimizer.zero_grad()
		loss_gen.backward()
		optimizer.step()
		
		# --- Phase 2: Train Discriminator ---
		
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
			print(f"Adv step [{i}] G_Loss: {loss_gen.item():.4f} D_Loss: {loss_disc.item():.4f}")