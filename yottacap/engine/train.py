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
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from yottacap.layer.yottacap import YottaCap
from yottacap.layer.loss import ASPLoss
from yottacap.config import Cfg
from yottacap.utils.chunker import NLTKChunker

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
		self.target_dicts = [None] * len(dataset)
		self.chunk_indices = [None] * len(dataset)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		item = self.dataset[index]
		if self.target_dicts[index] is not None:
			item['target_dict'] = self.target_dicts[index]
			item['chunk_indices'] = self.chunk_indices[index]
		return item

def custom_collate(batch):
	target_dicts = [item.pop('target_dict') for item in batch]
	batch = default_collate(batch)
	batch['target_dicts'] = target_dicts
	return batch

def precompute_embeddings(dataset_wrapper: YottaDatasetWrapper, clip_model: CLIP):
	print(f"Rank {Cfg.rank}: Pre-computing embeddings with NLTK Chunker...", flush=True)
	device = Cfg.device
	# Using larger batch size for faster encoding
	loader = DataLoader(dataset_wrapper.dataset, batch_size=256, num_workers=8, collate_fn=default_collate)
	chunker = NLTKChunker()
	
	global_idx = 0
	for batch in tqdm(loader, desc=f"Rank {Cfg.rank} Encoding"):
		text_embs = batch['text_emb'].to(device)
		B, _ = text_embs.shape
		
		batch_phrases = []
		batch_map = []
		batch_chunk_indices = []

		for b in range(B):
			seq = pad_tensor(text_embs[b], Cfg.max_seq_length, 0)
			
			# Real NLTK Chunking
			chunk_ends = chunker.get_chunk_ids(seq)
			batch_chunk_indices.append(chunk_ends.cpu())
			
			L = seq.shape[0]
			for i in range(L):
				if seq[i] == 0: continue
				
				# Always encode Word
				batch_phrases.append(seq[i:i+1])
				batch_map.append((b, i, i))
				
				# Encode Phrase only if different
				end = chunk_ends[i].item()
				if end > i:
					batch_phrases.append(seq[i:end+1])
					batch_map.append((b, i, end))
		
		# Batch Encode
		if batch_phrases:
			num_phrases = len(batch_phrases)
			clip_input = torch.zeros((num_phrases, Cfg.context_length), dtype=torch.long, device=device)
			for k, p in enumerate(batch_phrases):
				l = p.shape[0]
				clip_input[k, 0] = Cfg.sot_token_id
				clip_input[k, 1:1+l] = p
				clip_input[k, 1+l] = Cfg.eos_token_id
				
			with torch.no_grad():
				encoded = clip_model.encode_text(clip_input).float()
				encoded = encoded / encoded.norm(dim=-1, keepdim=True)
				encoded = encoded.cpu()
				
			for k, (b_idx, start, end) in enumerate(batch_map):
				abs_idx = global_idx + b_idx
				if dataset_wrapper.target_dicts[abs_idx] is None:
					dataset_wrapper.target_dicts[abs_idx] = {}
				dataset_wrapper.target_dicts[abs_idx][(start, end)] = encoded[k]
			
		for b in range(B):
			dataset_wrapper.chunk_indices[global_idx + b] = batch_chunk_indices[b]
			
		global_idx += B

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
	
	torch.cuda.set_device(Cfg.device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.cuda.manual_seed_all(42)
	
	wrapped_dataset = YottaDatasetWrapper(dataset)
	
	# Load CLIP and precompute
	clip_model, _ = clip.load(Cfg.clip_pretrained_path, device=Cfg.device, jit=False)
	clip_model.eval()
	precompute_embeddings(wrapped_dataset, clip_model)
	del clip_model
	torch.cuda.empty_cache()
	
	dist.barrier()
	
	yottacap_model = YottaCap()
	if init_weights is not None:
		yottacap_model.load_state_dict(torch.load(init_weights, map_location='cpu', weights_only=True))
	
	yottacap_model.to(Cfg.device)
	yottacap_model = DDP(module=yottacap_model, device_ids=[Cfg.rank], output_device=Cfg.rank)
	
	asp_loss_fn = ASPLoss()
	
	optimizer = AdamW(yottacap_model.parameters(), lr=lr)
	sampler = DistributedSampler(wrapped_dataset)
	dataloader = DataLoader(wrapped_dataset, sampler=sampler, batch_size=batch_size, 
							drop_last=True, num_workers=8, pin_memory=True, collate_fn=custom_collate)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Cfg.warmup_steps, num_training_steps=epochs * len(dataloader))
	
	for epoch in range(start_epoch, epochs + start_epoch):
		if Cfg.is_master:
			print(f">>> Training epoch {epoch}", flush=True)
			progress = tqdm(total = len(dataloader))
		dist.barrier()
		
		for item in dataloader:
			text_emb = item['text_emb'].to(Cfg.device, non_blocking=True)
			image_feat = item['image_feat'].to(Cfg.device, non_blocking=True)
			chunk_indices = item['chunk_indices'].to(Cfg.device, non_blocking=True)
			target_dicts = item['target_dicts']
			
			with torch.no_grad():
				clip_feature = image_feat / image_feat.norm(dim=-1, keepdim=True)
			
			token_ids = pad_tensor(text_emb, Cfg.max_seq_length, 1)
			
			features = yottacap_model.module.forward_hidden(clip_feature.float(), token_ids)
			preds = features[:, : -1, :] 
			
			loss = asp_loss_fn(preds, token_ids, chunk_indices, target_dicts)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			
			if Cfg.is_master:
				progress.set_postfix({'loss': loss.item()})
				progress.update()
			
		if Cfg.is_master:
			with open(log_file, 'a+') as f:
				f.writelines(f'epoch {epoch}: {progress.postfix}\n')
			progress.close()
			torch.save(yottacap_model.module.state_dict(), os.path.join(output_dir, f"{epoch:03d}.pt"))
			
	return yottacap_model