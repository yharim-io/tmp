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
from enum import Enum

# 必须从外部导入
from dataset import CocoDataset 

# 导入 clipcap 模块
from clipcap.layer import ClipCaptionModel, ClipCaptionPrefix, MappingType
from clipcap.config import Config

# --- 辅助函数 (严格复制自 decap/engine/train.py) ---

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

# --- 主训练函数 (命名遵循 decap) ---

def train(
	dataset: CocoDataset,
	output_dir: Path,
	log_dir: Path | None = None,
	report_gap: int = 10,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
	mapping_type: MappingType = MappingType.MLP,
	only_prefix: bool = False
):
	
	batch_size = Config.policy.batch_size
	lr = Config.policy.learning_rate
	
	os.makedirs(output_dir, exist_ok=True)
	if log_dir is not None:
		os.makedirs(log_dir, exist_ok=True)
		log_file = log_dir / get_time_now()
	
	# DDP 设置 (同 decap)
	device = torch.device(f'cuda:{Config.rank}')
	torch.cuda.set_device(device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.cuda.manual_seed_all(42)
	
	# --- 模型初始化 (clipcap 逻辑) ---
	if only_prefix:
		model = ClipCaptionPrefix(
			prefix_length=Config.model.prefix_length,
			clip_length=Config.model.prefix_length, # 假设 clip_length == prefix_length
			prefix_size=Config.model.clip_dim,
			num_layers=Config.model.num_layers,
			mapping_type=mapping_type
		)
		if Config.is_master: print("Training only prefix (mapper).")
	else:
		model = ClipCaptionModel(
			prefix_length=Config.model.prefix_length,
			clip_length=Config.model.prefix_length,
			prefix_size=Config.model.clip_dim,
			num_layers=Config.model.num_layers,
			mapping_type=mapping_type
		)
		if Config.is_master: print("Training both prefix (mapper) and GPT.")
	
	if init_weights is not None:
		model.load_state_dict(
			torch.load(
				init_weights,
				map_location=torch.device('cpu'),
				weights_only=True
			)
		)
	
	# CLIP (同 decap)
	clip_model, preprocess = clip.load(Config.model.clip_model_type, device=device, jit=False)
	clip_model.eval()
	
	# Loss (同 decap)
	loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
	
	model.to(device)
	model = DDP(model, device_ids=[Config.rank], output_device=Config.rank)
	
	# Optimizer (同 decap)
	optimizer = AdamW(model.parameters(), lr=lr)
	
	# Dataloader (同 decap, 使用 CocoDataset)
	sampler = DistributedSampler(dataset)
	dataloader = DataLoader(
		dataset=dataset,
		sampler=sampler,
		batch_size=batch_size,
		drop_last=True
	)
	
	# Scheduler (同 decap)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=Config.policy.warmup_steps,
		num_training_steps=epochs * len(dataloader)
	)
	
	# --- 训练循环 (严格遵循 decap 训练模式) ---
	
	for epoch in range(start_epoch, epochs + start_epoch):
		
		loss_token_save, ac_token_save = 0, 0
		sys.stdout.flush()
		
		if Config.is_master:
			print(f">>> Training epoch {epoch}")
			progress = tqdm(total=len(dataloader)//report_gap)
		
		dist.barrier()
		
		# 关键：使用 decap 的数据循环模式 (来自 CocoDataset)
		for idx, caption in enumerate(dataloader):
			
			# 1. 使用 clip.tokenize (decap 模式)
			token_ids_77 = clip.tokenize(caption).clone().detach().long().to(device)
			
			# 2. 获取文本前缀 (decap 模式)
			with torch.no_grad():
				feature_text = clip_model.encode_text(token_ids_77)
				feature_text /= feature_text.norm(dim=-1, keepdim=True)
			
			# 3. 获取目标 token (decap 模式)
			token_ids = pad_tensor(token_ids_77, Config.model.max_seq_length, 1).to(device)
			
			# 4. Forward (调用 clipcap model)
			# model.forward(self, tokens: torch.Tensor, prefix: torch.Tensor, ...)
			outputs = model(tokens=token_ids, prefix=feature_text.float())
			logits = outputs.logits # 我们在 clipcap.py 中确保了 .logits 的存在
			logits = logits[:, : -1]
			
			# 5. 计算 Loss (decap 模式)
			token_ids_flat = token_ids.flatten()
			logits_flat = logits.reshape(-1, logits.shape[-1])
			
			loss_token = loss_ce(logits_flat, token_ids_flat)
			ac_token = ((logits_flat.argmax(1) == token_ids_flat) * (token_ids_flat > 0)).sum() / (token_ids_flat > 0).sum()
			
			optimizer.zero_grad()
			loss_all = loss_token
			loss_all.backward()
			optimizer.step()
			scheduler.step()
			
			# 日志记录 (同 decap)
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
		
		# 保存 (同 decap)
		if Config.is_master:
			if log_dir is not None:
				with open(log_file, 'a+') as f:
					f.writelines(f'epoch {epoch}: {progress.postfix}\n')
			progress.close()
			torch.save(
				model.module.state_dict(),
				os.path.join(output_dir, f"{epoch:03d}.pt")
			)
			
	return model
