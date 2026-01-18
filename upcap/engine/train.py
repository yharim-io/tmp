import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from upcap.model.upcap import UpCap
from upcap.config import Cfg

def get_time_now() -> str:
	now = datetime.now()
	return now.strftime("%Y-%m-%d_%H:%M")

def train(
	dataset: Dataset,
	collate_fn,
	output_dir: Path,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
) -> UpCap:
	
	batch_size = Cfg.batch_size
	lr = Cfg.learning_rate
	
	log_dir = output_dir / 'log/'
	
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	log_file = log_dir / get_time_now()
	
	torch.cuda.set_device(Cfg.device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.cuda.manual_seed_all(42)
	torch.manual_seed(42)
	
	upcap_model = UpCap()
	
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
		output_device=Cfg.rank
	)
	
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
		num_warmup_steps=1000,
		num_training_steps=epochs * len(dataloader)
	)
	
	loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		
		if Cfg.is_master:
			print(f">>> Training epoch {epoch}", flush=True)
			progress = tqdm(total = len(dataloader))
		
		dist.barrier()
		
		for batch in dataloader:
			
			text_concepts: Tensor = batch['text_concepts'].to(Cfg.device, non_blocking=True)
			token_ids: Tensor = batch['token_ids'].to(Cfg.device, non_blocking=True)
			
			M = text_concepts.shape[1]
			L = token_ids.shape[1]
			V = upcap_model.module.gpt2.emb_size 
			
			logits = upcap_model(text_concepts, token_ids)
			
			logits_for_loss = logits[:, M-1 : -1, :]
			
			loss = loss_ce(logits_for_loss.reshape(-1, logits.shape[-1]), token_ids.reshape(-1))
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			
			if Cfg.is_master:
				progress.set_postfix({
					'loss': loss.item()
				})
				progress.update()
			
		if Cfg.is_master:
			with open(log_file, 'a+') as f:
				f.writelines(f'epoch {epoch}: {progress.postfix}\n')
			progress.close()
			torch.save(
				upcap_model.module.state_dict(),
				os.path.join(output_dir, f"{epoch:03d}.pt")
			)
			
	return upcap_model