import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import clip
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from decap.layer.decap import DeCap
from decap.config import Cfg

def get_time_now() -> str:
	now = datetime.now()
	return now.strftime("%Y-%m-%d_%H:%M")

def pad_tensor(tensor: Tensor, max_len: int, dim: int) -> Tensor:
	current_len: int = tensor.shape[dim]
	padding: int = max_len - current_len

	if padding > 0:
		pad_shape = list(tensor.shape)
		pad_shape[dim] = padding
		pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
		tensor = torch.cat([tensor, pad_tensor], dim=dim)
	elif padding < 0:
		tensor = tensor.narrow(dim, 0, max_len)

	return tensor

def train_text_only(
	dataset: Dataset,
	output_dir: Path,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
) -> DeCap:
	
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
	
	decap_model = DeCap()
	
	if init_weights is not None:
		decap_model.load_state_dict(
			torch.load(
				init_weights,
				map_location=torch.device('cpu'),
				weights_only=True
			)
		)
	
	clip_model, preprocess = clip.load(Cfg.clip_pretrained_path, device=Cfg.device, jit=False)
	clip_model.eval()
	
	decap_model.to(Cfg.device)
	decap_model = DDP(
		module=decap_model,
		device_ids=[Cfg.rank],
		output_device=Cfg.rank
	)
	
	optimizer = AdamW(decap_model.parameters(), lr=lr)
	
	sampler = DistributedSampler(dataset)
	
	dataloader = DataLoader(
		dataset=dataset,
		sampler=sampler,
		batch_size=batch_size,
		drop_last=True,
		num_workers=8,
		pin_memory=True
	)
	
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=Cfg.warmup_steps,
		num_training_steps=epochs * len(dataloader)
	)
	
	loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		
		if Cfg.is_master:
			print(f">>> Training epoch {epoch}", flush=True)
			progress = tqdm(total = len(dataloader))
		
		dist.barrier()
		
		for item in dataloader:
			
			text_emb: Tensor = item['text_emb']
			text_emb = text_emb.to(Cfg.device, non_blocking=True)
			
			with torch.no_grad():
				clip_feature = clip_model.encode_text(text_emb)
				clip_feature /= clip_feature.norm(dim=-1, keepdim=True)
			
			token_ids = pad_tensor(text_emb, Cfg.max_seq_length, 1)
			logits = decap_model(clip_feature.float(), token_ids)
			logits = logits[:, : -1]
			
			token_ids = token_ids.flatten()
			logits = logits.reshape(-1, logits.shape[-1])
			
			loss_token = loss_ce(logits, token_ids)
			ac_token = ((logits.argmax(1)==token_ids) * (token_ids>0)).sum() / (token_ids>0).sum()
			
			optimizer.zero_grad()
			loss_all = loss_token
			loss_all.backward()
			optimizer.step()
			scheduler.step()
			
			if Cfg.is_master:
				progress.set_postfix({
					'loss_token': loss_token.item(),
					'ac_token': ac_token.item()
				})
				progress.update()
			
		if Cfg.is_master:
			with open(log_file, 'a+') as f:
				f.writelines(f'epoch {epoch}: {progress.postfix}\n')
			progress.close()
			torch.save(
				decap_model.module.state_dict(),
				os.path.join(output_dir, f"{epoch:03d}.pt")
			)
			
	return decap_model
