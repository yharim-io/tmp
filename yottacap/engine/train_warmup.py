import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim import AdamW
from tqdm import tqdm

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg

def train_warmup(
	dataloader,
	model: YottaCap,
):
	model.train()
	ce_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	
	def set_freeze(modules: list[nn.Module], freeze: bool):
		for mod in modules:
			for p in mod.parameters():
				p.requires_grad = not freeze
	
	set_freeze([model.image_adapter, model.discriminator], True)
	set_freeze([model.text_adapter], False)
	main_params = [
		p for n, p in model.named_parameters()
		if 'image_adapter' not in n and 'discriminator' not in n
	]
	optimizer: Optimizer = AdamW(main_params, lr=Cfg.learning_rate)

	tqdmloader = tqdm(dataloader, desc='Text Priming') if Cfg.is_master else dataloader
	
	for batch in tqdmloader:
		text_emb: Tensor = batch['text_emb']
		text_emb = text_emb.to(Cfg.device, non_blocking=True)
		
		features = model.extract_clip_features(text=text_emb)
		if 'text_tokens' not in features:
			continue
			
		S_text: Tensor = model.get_text_latent(features['text_tokens'])
		
		# CE LOSS
		logits = model.forward(S_text, text_emb[:, :-1])
		logits = logits[:, S_text.shape[1]:]
		loss_ce = ce_loss_fn(logits.reshape(-1, logits.shape[-1]), text_emb[:, 1:].flatten())
		
		# KL LOSS
		loss_kl = S_text.pow(2).mean()
		
		loss_text = loss_ce + Cfg.kl_weight * loss_kl
		
		optimizer.zero_grad()
		loss_text.backward()
		optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'LossText': loss_text.item(),
				'LossCE': loss_ce.item(),
				'LossKL': loss_kl.item(),
			})

	set_freeze([model.text_adapter, model.discriminator], True)
	set_freeze([model.image_adapter], False)
	main_params = [
		p for n, p in model.named_parameters()
		if 'text_adapter' not in n and 'discriminator' not in n
	]
	optimizer: Optimizer = AdamW(main_params, lr=Cfg.learning_rate)

	tqdmloader = tqdm(dataloader, desc='Image Alignment') if Cfg.is_master else dataloader

	for batch in tqdmloader:
		image_emb = batch['image_emb']
		image_emb = image_emb.to(Cfg.device, non_blocking=True)
		
		S_img: Tensor = model.get_image_latent(image_emb)
		T_image: Tensor = batch['image_feat']
		
		# SIM LOSS
		prefix = model.latent_proj(S_img)
		logits_img = model.gpt2.forward_logits(prefix)
		soft_embeds = model.gumbel_softmax(logits_img)
		
		T_text_recon: Tensor = model.project_to_clip(soft_embeds)
		T_text_recon = F.normalize(T_text_recon, dim=-1)
		
		loss_sim = 1.0 - (T_text_recon * T_image).sum(dim=-1).mean()
		
		# KL LOSS
		loss_kl = S_img.pow(2).mean()
			
		loss_img = loss_sim + Cfg.kl_weight * loss_kl
		
		optimizer.zero_grad()
		loss_img.backward()
		optimizer.step()
		
		if Cfg.is_master:
			tqdmloader.set_postfix({
				'LossImg': loss_img.item(),
				'LossSim': loss_sim.item(),
				'LossKL': loss_kl.item(),
			})
	
	set_freeze([model.text_adapter, model.image_adapter], True)
	set_freeze([model.discriminator], False)
	optimizer: Optimizer = AdamW(model.discriminator.parameters(), lr=Cfg.learning_rate)
	
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