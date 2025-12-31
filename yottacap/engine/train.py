import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import clip
from clip.model import CLIP
import os
import gc
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import nullcontext

from yottacap.layer.yottacap import YottaCap
from yottacap.layer.loss import ASPLoss
from yottacap.config import Cfg
from yottacap.utils.chunker import NLTKChunker
from utils.logger import logger

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

class YottaDatasetWrapper(Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
		# 扁平化存储结构
		self.embeddings = torch.empty(0) # (Total_Targets, Dim)
		self.metadata = torch.empty(0)   # (Total_Targets, 2) -> [start, end]
		self.offsets = torch.zeros(1, dtype=torch.long) # 索引指针
		
		self.chunk_indices = [None] * len(dataset)
		self.has_cache = False

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		item = self.dataset[index]
		
		if self.has_cache:
			# 仅在需要时动态构建 dict，底层存储为 Tensor
			start_idx = self.offsets[index].item()
			end_idx = self.offsets[index + 1].item()
			
			if start_idx < end_idx:
				sample_meta = self.metadata[start_idx:end_idx] # (K, 2)
				sample_embs = self.embeddings[start_idx:end_idx] # (K, Dim)
				
				target_dict = {}
				# 这里的循环次数很少（每个样本只有几个短语），速度很快
				for i in range(sample_meta.shape[0]):
					s, e = sample_meta[i, 0].item(), sample_meta[i, 1].item()
					target_dict[(s, e)] = sample_embs[i]
				
				item['target_dict'] = target_dict
			else:
				item['target_dict'] = {}
				
			item['chunk_indices'] = self.chunk_indices[index]
			
		return item

	def load_cache(self, embeddings: Tensor, metadata: Tensor, chunk_indices: list):
		"""
		metadata: Tensor (Total_Targets, 3) -> [sample_idx, start, end]
		"""
		self.embeddings = embeddings
		self.chunk_indices = chunk_indices
		
		# 构建索引加速查找
		# metadata[:, 0] 是 sample_idx，利用 bincount 计算每个样本有多少个 target
		num_samples = len(self.dataset)
		sample_ids = metadata[:, 0].long()
		
		# 只保留 start, end
		self.metadata = metadata[:, 1:].clone()
		
		counts = torch.bincount(sample_ids, minlength=num_samples)
		self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, dim=0)])
		self.has_cache = True

def custom_collate(batch):
	target_dicts = [item.pop('target_dict') for item in batch]
	batch = default_collate(batch)
	batch['target_dicts'] = target_dicts
	return batch

def precompute_embeddings(
	dataset_wrapper: YottaDatasetWrapper, 
	cache_dir: Path, 
	clip_model: CLIP, 
	device: torch.device, 
	is_master: bool
):
	cache_file = cache_dir / "tensor_cache.pt"
	
	if cache_file.exists():
		if is_master:
			print(f"[Precompute] Found existing tensor cache in {cache_file}.")
	else:
		if is_master:
			with logger("Precompute", "Computing (Tensor Mode) on Rank 0"):
				clip_model.eval()
				loader = DataLoader(
					dataset_wrapper.dataset,
					batch_size=512,
					num_workers=0,
					collate_fn=default_collate,
					shuffle=False
				)
				chunker = NLTKChunker()
				
				all_emb_list = []
				all_meta_list = [] # [sample_idx, start, end]
				all_chunk_list = []
				
				global_idx = 0
				
				for batch in tqdm(loader, desc="Encoding"):
					text_embs = batch['text_emb']
					B, _ = text_embs.shape
					
					batch_phrases = []
					batch_map = [] # (batch_local_idx, start, end)
					batch_chunk_ends = []

					for b in range(B):
						seq = pad_tensor(text_embs[b], Cfg.max_seq_length, 0)
						chunk_ends = chunker.get_chunk_ids(seq)
						# 将 chunk_indices 转为 Tensor 统一存储
						batch_chunk_ends.append(chunk_ends)
						
						L = seq.shape[0]
						for i in range(L):
							if seq[i] == 0: continue
							batch_phrases.append(seq[i:i+1])
							batch_map.append((b, i, i))
							end = chunk_ends[i].item()
							if end > i:
								batch_phrases.append(seq[i:end+1])
								batch_map.append((b, i, end))
					
					# 处理 CLIP
					if batch_phrases:
						num_phrases = len(batch_phrases)
						clip_input = torch.zeros((num_phrases, Cfg.context_length), dtype=torch.long)
						for k, p in enumerate(batch_phrases):
							l = p.shape[0]
							clip_input[k, 0] = Cfg.sot_token_id
							clip_input[k, 1:1+l] = p
							clip_input[k, 1+l] = Cfg.eos_token_id
						
						clip_input = clip_input.to(device)
						with torch.no_grad():
							encoded = clip_model.encode_text(clip_input).float()
							encoded = encoded / encoded.norm(dim=-1, keepdim=True)
							encoded = encoded.cpu()
						
						all_emb_list.append(encoded)
						
						# 构建元数据 [sample_global_idx, start, end]
						meta_tensor = torch.tensor(batch_map, dtype=torch.long)
						meta_tensor[:, 0] += global_idx # 修正为全局样本ID
						all_meta_list.append(meta_tensor)
					
					all_chunk_list.extend(batch_chunk_ends)
					global_idx += B
				
				data_to_save = {
					'embeddings': torch.cat(all_emb_list) if all_emb_list else torch.empty(0),
					'metadata': torch.cat(all_meta_list) if all_meta_list else torch.empty(0),
					'chunks': torch.stack(all_chunk_list)
				}
				torch.save(data_to_save, cache_file)
				
				del all_emb_list, all_meta_list, all_chunk_list
				gc.collect()

	dist.barrier(device_ids=[device.index])
	
	with logger(f"Rank {Cfg.rank}", "Loading Tensor Cache"):
		gc.collect()
		
		data = torch.load(cache_file, map_location='cpu', weights_only=True)
		final_chunks_list = [t for t in data['chunks']] 
		
		dataset_wrapper.load_cache(data['embeddings'], data['metadata'], final_chunks_list)
		
		del data
		gc.collect()

	dist.barrier(device_ids=[device.index])

def train(
	dataset: Dataset,
	output_dir: Path,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
) -> YottaCap:
	
	batch_size = Cfg.batch_size
	lr = Cfg.learning_rate
	log_dir = output_dir / 'log/'
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	log_file = log_dir / get_time_now()
	
	dist.init_process_group(
		backend='nccl', 
		init_method='env://', 
		timeout=timedelta(hours=5)
	)
	local_rank = int(os.environ['LOCAL_RANK'])
	global_rank = dist.get_rank()
	is_master = (global_rank == 0)

	device = torch.device(f'cuda:{local_rank}')
	torch.cuda.set_device(device)
	torch.cuda.manual_seed_all(42)
	
	if is_master:
		print(f"[Init] Global Rank: {global_rank}, Device: {device}")

	clip_model, _ = clip.load(Cfg.clip_pretrained_path, device=device, jit=False)
	clip_model.eval()

	wrapped_dataset = YottaDatasetWrapper(dataset)
	
	precompute_embeddings(wrapped_dataset, output_dir, clip_model, device, is_master)
	
	dist.barrier(device_ids=[local_rank])
	
	yottacap_model = YottaCap()
	if init_weights is not None:
		yottacap_model.load_state_dict(torch.load(init_weights, map_location='cpu', weights_only=True))
	
	yottacap_model.to(device)
	yottacap_model = DDP(module=yottacap_model, device_ids=[local_rank], output_device=local_rank)
	
	ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
	# asp_loss_fn = ASPLoss()
	
	optimizer = AdamW(yottacap_model.parameters(), lr=lr)
	sampler = DistributedSampler(wrapped_dataset)
	
	dataloader = DataLoader(
		wrapped_dataset,
		sampler=sampler,
		batch_size=batch_size,
		drop_last=True,
		num_workers=0,
		pin_memory=True,
		collate_fn=custom_collate
	)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=Cfg.warmup_steps,
		num_training_steps=epochs * len(dataloader)
	)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		sampler.set_epoch(epoch)
		
		if is_master:
			log_ctx = logger("Train", f"Epoch {epoch}")
		else:
			log_ctx = nullcontext()

		with log_ctx:
			if is_master:
				progress = tqdm(total = len(dataloader))
			
			for item in dataloader:
				text_emb = item['text_emb'].to(device, non_blocking=True)
				chunk_indices = item['chunk_indices'].to(device, non_blocking=True)
				target_dicts = item['target_dicts']
				
				chunk_indices = chunk_indices[:, 1:]
				
				with torch.no_grad():
					clip_feature = clip_model.encode_text(text_emb)
					clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
					clip_feature = clip_feature.float()
				
				flat_targets = []
				map_info = []
				for b_idx, t_dict in enumerate(target_dicts):
					for k, v in t_dict.items():
						flat_targets.append(v)
						map_info.append((b_idx, k))
				
				if flat_targets:
					flat_targets_tensor = torch.stack(flat_targets).to(device)
					projected_targets = yottacap_model.module.mlp(flat_targets_tensor)
					
					for i, (b_idx, k) in enumerate(map_info):
						target_dicts[b_idx][k] = projected_targets[i]

				token_ids = pad_tensor(text_emb, Cfg.max_seq_length, 1)
				
				hidden_states, logits = yottacap_model.module.forward(clip_feature, token_ids)
				hidden_states = hidden_states[:, :-1, :]
				logits = logits[:, :-1, :]
				
				ce_loss = ce_loss_fn(
					logits.reshape(-1, logits.shape[-1]),
					token_ids.reshape(-1)
				)
				
				# asp_loss = asp_loss_fn(
				# 	hidden_states,
				# 	token_ids,
				# 	chunk_indices,
				# 	target_dicts
				# )
				
				# loss = ce_loss * Cfg.asp_temperature + asp_loss

				loss = ce_loss
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()
				
				if is_master:
					progress.set_postfix({
						'loss': loss.item(),
						# 'ce': ce_loss.item(),
						# 'asp': asp_loss.item()
					})
					progress.update()
			
			if is_master:
				with open(log_file, 'a+') as f:
					f.writelines(f'epoch {epoch}: {progress.postfix}\n')
				progress.close()
				torch.save(yottacap_model.module.state_dict(), os.path.join(output_dir, f"{epoch:03d}.pt"))

	dist.destroy_process_group()
	
	return yottacap_model