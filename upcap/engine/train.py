import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
from pathlib import Path
import clip

from upcap.model.upcap import UpCap
from upcap.config import Cfg
from utils.tool import tqdm, get_time_now
from utils.dist import dist_startup

@dist_startup()
def train(
	dataset: Dataset,
	collate_fn,
	output_dir: Path,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
	lr: float | None = None,
	warmup_steps: int | None = None,
) -> UpCap:
	
	batch_size = Cfg.batch_size
	if lr is None: lr = Cfg.learning_rate
	if warmup_steps is None: warmup_steps = Cfg.warmup_steps
	
	if Cfg.is_master:
		log_dir = output_dir / 'log/'
		os.makedirs(log_dir, exist_ok=True)
		log_file = log_dir / get_time_now()
	
	upcap_model = UpCap(
		enable_concepts_global_buffer=True,
		enable_concepts_local_buffer=True,
	)
	
	if init_weights is not None:
		upcap_model.load_state_dict(
			torch.load(
				init_weights,
				map_location=torch.device('cpu'),
				weights_only=True
			)
		)
	
	upcap_model.to(Cfg.device)
	upcap_model = DDP(
		module=upcap_model,
		device_ids=[Cfg.rank],
		output_device=Cfg.rank,
		find_unused_parameters=True
	)

	clip_model, _ = clip.load(Cfg.clip_pretrained_path, device=Cfg.device, jit=False)
	clip_model.eval()
	
	with torch.no_grad():
		zero_token = torch.zeros((1, 77), dtype=torch.long, device=Cfg.device)
		pad_feat = clip_model.encode_text(zero_token).float()
		pad_feat /= pad_feat.norm(dim=-1, keepdim=True)

	optimizer = AdamW(upcap_model.parameters(), lr=lr)
	
	sampler = DistributedSampler(dataset)
	
	dataloader = DataLoader(
		dataset=dataset,
		sampler=sampler,
		batch_size=batch_size,
		drop_last=True,
		num_workers=8,
		pin_memory=True,
		collate_fn=collate_fn
	)
	
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=epochs * len(dataloader)
	)
	
	loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		
		if Cfg.is_master:
			print(f">>> Training epoch {epoch}", flush=True)
			progress = tqdm(total = len(dataloader))
		
		sampler.set_epoch(epoch)
		dist.barrier()
		
		for batch in dataloader:
			
			text_concept_tokens_list: list[Tensor] = batch['text_concept_tokens']
			token_ids: Tensor = batch['token_ids'].to(Cfg.device, non_blocking=True)
			
			counts = [len(t) for t in text_concept_tokens_list]
			flat_tokens = torch.cat(text_concept_tokens_list, dim=0).to(Cfg.device, non_blocking=True)

			with torch.no_grad():
				flat_feats = clip_model.encode_text(flat_tokens)
				flat_feats /= flat_feats.norm(dim=-1, keepdim=True)
				feats_list = flat_feats.split(counts)
			
			batch_global_feat = []
			batch_local_feat = []
			
			for feats in feats_list:
				global_c = feats[0:1]
				local_c = feats[1:]

				if local_c.shape[0] > 0:
					sim = (local_c @ global_c.transpose(-2, -1)).squeeze(-1)
					local_c = local_c[sim.argsort(descending=True)]
					local_c = local_c[:Cfg.max_concepts - 1]
				
				n_pad = Cfg.max_concepts - 1 - local_c.shape[0]
				if n_pad > 0:
					local_c = torch.cat([local_c, pad_feat.expand(n_pad, -1)], dim=0)
				
				batch_global_feat.append(global_c)
				batch_local_feat.append(local_c)
			
			global_feat = torch.stack(batch_global_feat).float()
			local_feat = torch.stack(batch_local_feat).float()
			
			global_emb, local_emb = upcap_model.module.project_features(
				global_feat, local_feat, global_attn=True, local_attn=True
			)
			text_emb = upcap_model.module.embed_tokens(token_ids)
			
			inputs_embeds, cross_states = upcap_model.module.assemble_structure(
				global_emb, local_emb, text_emb
			)
			
			logits, _ = upcap_model(inputs_embeds, cross_states)
			logits = logits[:, :-1]
			
			token_ids = token_ids.flatten()
			logits = logits.reshape(-1, logits.shape[-1])
			
			loss_token = loss_ce(logits, token_ids)
			ac_token = ((logits.argmax(1) == token_ids) * (token_ids > 0)).sum() / (token_ids > 0).sum()
			
			optimizer.zero_grad()
			loss_all = loss_token
			loss_all.backward()
			optimizer.step()
			scheduler.step()
			
			last_lr = 0
			current_lr = scheduler.get_last_lr()[0]
			if current_lr != 0:
				last_lr = current_lr

			if Cfg.is_master:
				progress.set_postfix({
					'loss_token': loss_token.item(),
					'ac_token': ac_token.item(),
					'lr': last_lr
				})
				progress.update()
			
		if Cfg.is_master:
			with open(log_file, 'a+') as f:
				f.writelines(f'epoch {epoch}: {progress.postfix}\n')
			progress.close()
			torch.save(
				upcap_model.module.state_dict(),
				output_dir / f"{epoch:03d}.pt"
			)

	return upcap_model