import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import clip
import sys, os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from dataset import CocoDataset
from ..layer import DeCap
from ..config import Config

def get_time_now() -> str:
	now = datetime.now()
	return now.strftime("%Y-%m-%d_%H:%M")

def pad_tensor(tensor: Tensor, max_len: int, dim: int) -> Tensor:
	current_len = tensor.shape[dim]
	padding = max_len - current_len

	if padding > 0:
		pad_shape = list(tensor.shape)
		pad_shape[dim] = padding
		pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
		tensor = torch.cat([tensor, pad_tensor], dim=dim)
	elif padding < 0:
		tensor = tensor.narrow(dim, 0, max_len)

	return tensor

def train(
	dataset: CocoDataset,
	output_dir: Path,
	log_dir: Path | None = None,
	report_gap: int = 10,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
):
	
	batch_size = Config.policy.batch_size
	lr = Config.policy.learning_rate
	
	os.makedirs(output_dir, exist_ok=True)
	if log_dir is not None:
		os.makedirs(log_dir, exist_ok=True)
		log_file = log_dir / get_time_now()
	
	device = torch.device(f'cuda:{Config.rank}')
	torch.cuda.set_device(device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.cuda.manual_seed_all(42)
	
	decap_model = DeCap()
	
	if init_weights is not None:
		decap_model.load_state_dict(
			torch.load(
				init_weights,
				map_location=torch.device('cpu'),
				weights_only=True
			)
		)
	
	clip_model, preprocess = clip.load(Config.model.clip_model_type, device=device, jit=False)
	clip_model.eval()
	
	loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
	decap_model.to(device)
	decap_model = DDP(decap_model, device_ids=[Config.rank], output_device=Config.rank)
	
	optimizer = AdamW(decap_model.parameters(), lr=lr)
	
	sampler = DistributedSampler(dataset)
	dataloader = DataLoader(
		dataset=dataset,
		sampler=sampler,
		batch_size=batch_size,
		drop_last=True
	)
	
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=Config.policy.warmup_steps,
		num_training_steps=epochs * len(dataloader)
	)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		
		loss_token_save, ac_token_save = 0, 0
		
		sys.stdout.flush()
		
		if Config.is_master:
			print(f">>> Training epoch {epoch}")
			progress = tqdm(total=len(dataloader)//report_gap)
		
		dist.barrier()
		
		for idx, caption in enumerate(dataloader):
			
			token_ids_77 = clip.tokenize(caption).clone().detach().long().to(device)
			
			with torch.no_grad():
				feature_text = clip_model.encode_text(token_ids_77)
				feature_text /= feature_text.norm(dim=-1, keepdim=True)
			
			token_ids = pad_tensor(token_ids_77, Config.model.max_seq_length, 1).to(device)
			logits = decap_model(feature_text.float(), token_ids)
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
			
			if Config.is_master:
				if (idx + 1) % report_gap == 0:
					progress.set_postfix({
						'loss_token': loss_token_save / report_gap,
						'ac_token': ac_token_save / report_gap
					})
					progress.update()
					loss_token_save, ac_token_save = 0, 0
				else:
					loss_token_save += loss_token.item()
					ac_token_save += ac_token.item()
			
		if Config.is_master:
			if log_dir is not None:
				with open(log_file, 'a+') as f:
					f.writelines(f'epoch {epoch}: {progress.postfix}\n')
			progress.close()
			torch.save(
				decap_model.module.state_dict(),
				os.path.join(output_dir, f"{epoch:03d}.pt")
			)
			
	return decap_model
