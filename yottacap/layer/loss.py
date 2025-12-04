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
		input_ids: Tensor,			# (B, L + 1)
		chunk_end_ids: Tensor,		# (B, L)
		target_dicts: list[dict]	# (B)
	) -> Tensor:
		
		B, L, D = pred_embeddings.shape
		dtype = pred_embeddings.dtype
		
		target_phrase_list = []
		mask_list = []
		
		input_ids_cpu = input_ids.cpu()
		chunk_end_cpu = chunk_end_ids.cpu()
		
		for b in range(B):
			dct = target_dicts[b]
			b_input_ids = input_ids_cpu[b]
			b_chunk_ends = chunk_end_cpu[b]
			
			b_targets = torch.zeros((L, D), device=Cfg.device, dtype=dtype)
			b_mask = torch.zeros((L,), device=Cfg.device, dtype=torch.bool)
			
			for t in range(L):
				curr_id = t + 1
				token_id = b_input_ids[curr_id].item()
				
				if token_id == 0:
					continue
				
				phrase_end_id = b_chunk_ends[t].item()
				
				key = (curr_id, phrase_end_id)
				if key in dct:
					b_targets[t] = dct[key]
					b_mask[t] = True
			
			target_phrase_list.append(b_targets)
			mask_list.append(b_mask)			
		
		target_phrase_embeds = torch.stack(target_phrase_list) # (B, L, D)
		
		pred_norm = F.normalize(pred_embeddings, p=2, dim=-1)
		target_norm = F.normalize(target_phrase_embeds, p=2, dim=-1)
		
		s_phrase = (pred_norm * target_norm).sum(dim=-1)
		
		mask = torch.stack(mask_list)
		asp_loss_raw = 1.0 - s_phrase
		
		if mask.sum() > 0:
			asp_loss = (asp_loss_raw * mask).sum() / mask.sum()
		else:
			asp_loss = torch.tensor(0.0, device=Cfg.device)
		
		return asp_loss