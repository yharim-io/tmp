import torch
from torch import nn, Tensor
from torch.nn import functional as F

from yottacap.config import Cfg

class ASPLoss(nn.Module):
    
	def __init__(self):
		super().__init__()
	
	def forward(
		self,
		pred_embeddings: Tensor,
		input_ids: Tensor,
		chunk_end_ids: Tensor,
		target_dicts: list[dict]
	) -> Tensor:
		
		B, L, D = pred_embeddings.shape
		pred_embeddings = F.normalize(pred_embeddings, p=2, dim=-1)

		target_word_list = []
		target_phrase_list = []

		chunk_end_cpu = chunk_end_ids.cpu().numpy()

		for b in range(B):
			dct = target_dicts[b]
			ends = chunk_end_cpu[b]

			b_word_embs = torch.zeros((L, D), device=Cfg.device)
			b_phrase_embs = torch.zeros((L, D), device=Cfg.device)

			for i in range(L):
				if (i, i) in dct:
					b_word_embs[i] = dct[(i, i)]
					b_phrase_embs[i] = dct[(i, ends[i])]

			target_word_list.append(b_word_embs)
			target_phrase_list.append(b_phrase_embs)
		
		target_word_embs = torch.stack(target_word_list)
		
		s_word = (pred_embeddings * target_word_embs).sum(dim=-1)

		is_same = (chunk_end_ids == torch.arange(L, device=Cfg.device).unsqueeze(0))
		mask_pad = (input_ids != 0)
		
		final_loss = torch.zeros_like(s_word)

		mask_a = is_same & mask_pad
		if mask_a.any():
			final_loss[mask_a] = 1.0 - Cfg.asp_temperature * Cfg.ln2 - s_word[mask_a]
		
		mask_b = (~ is_same) & mask_pad
		if mask_b.any():
			target_phrase_embs = torch.stack(target_phrase_list)
			s_phrase_part = (pred_embeddings[mask_b] * target_phrase_embs[mask_b]).sum(dim=-1)
			s_word_part = s_word[mask_b]
			term = torch.exp(s_word_part / Cfg.asp_temperature) \
				 + torch.exp(s_phrase_part / Cfg.asp_temperature)
			final_loss[mask_b] = 1.0 - Cfg.asp_temperature * torch.log(term)
		
		return final_loss.sum() / mask_pad.sum().clamp(min=1.0)