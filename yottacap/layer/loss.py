import torch
from torch import nn, Tensor
from torch.nn import functional as F
from yottacap.config import Cfg

class ASPLoss(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
	
	def forward(
		self,
		pred_embeddings: Tensor,	# (B, L, D)
		logits: Tensor,				# (B, L, V)
		input_ids: Tensor,			# (B, L + 1)
		chunk_end_ids: Tensor,		# (B, L)
		target_dicts: list[dict]	# (B)
	) -> Tensor:
		
		target_ids = input_ids[:, 1:]	# (B, L)
		
		ce_loss = self.ce_loss(
			logits.reshape(-1, logits.shape[-1]),
			target_ids.reshape(-1)
		)
		
		B, L, D = pred_embeddings.shape
		dtype = pred_embeddings.dtype
		
		target_phrase_list = []
		input_ids_cpu = input_ids.cpu()
		chunk_end_cpu = chunk_end_ids.cpu()
		
		for b in range(B):
			dct = target_dicts[b]
			b_input_ids = input_ids_cpu[b]
			b_chunk_ends = chunk_end_cpu[b]
			
			b_targets = torch.zeros((L, D), device=Cfg.device, dtype=dtype)
			
			for t in range(L):
				token_id = b_input_ids[t + 1].item()
				
				if token_id == 0:
					continue
				
				phrase_end_pos = b_chunk_ends[t]
				phrase_end_id = b_input_ids[phrase_end_pos].item()
				
				key = (token_id, phrase_end_id)
				if key in dct:
					b_targets[t] = dct[key]
			
			target_phrase_list.append(b_targets)
		
		target_phrase_embeds = torch.stack(target_phrase_list) # (B, L, D)
		
		pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)
		target_norm = F.normalize(target_phrase_embeds, p=2, dim=-1)
		
		s_phrase = (pred_norm * target_norm).sum(dim=-1)
		
		mask = (target_ids != 0)
		sim_loss_raw = 1.0 - s_phrase
		
		if mask.sum() > 0:
			sim_loss = (sim_loss_raw * mask).sum() / mask.sum()
		else:
			sim_loss = torch.tensor(0.0, device=Cfg.device)
		
		return ce_loss * Cfg.asp_temperature + sim_loss