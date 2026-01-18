import torch
from torch import Tensor
import clip
from upcap.config import Cfg
from utils.dataset import CocoDataset, DType
from upcap.engine.train import train
from upcap.model.parser import TextParser

class CollateFn:
	def __init__(self):
		self.parser = TextParser()
		self.clip_model, _ = clip.load(Cfg.clip_pretrained_path, device='cpu', jit=False)
		self.clip_model.eval()

	def __call__(self, batch):
		texts = [item['text'] for item in batch]
		
		all_concepts = []
		all_tokens = []
		
		for text in texts:
			concepts = self.parser(text)
			
			concept_tokens = clip.tokenize(concepts, truncate=True)
			with torch.no_grad():
				concept_feats = self.clip_model.encode_text(concept_tokens)
				concept_feats = concept_feats / concept_feats.norm(dim=-1, keepdim=True)
			all_concepts.append(concept_feats)
			
			tokens = clip.tokenize(text, truncate=True).squeeze(0)
			all_tokens.append(tokens)

		padded_concepts = torch.zeros(len(batch), Cfg.max_concepts, Cfg.clip_dim)
		padded_tokens = torch.zeros(len(batch), Cfg.max_seq_length, dtype=torch.long)

		for i, (c, t) in enumerate(zip(all_concepts, all_tokens)):
			n_c = min(c.shape[0], Cfg.max_concepts)
			padded_concepts[i, :n_c] = c[:n_c]
			
			n_t = min(t.shape[0], Cfg.max_seq_length)
			padded_tokens[i, :n_t] = t[:n_t]

		return {
			'text_concepts': padded_concepts,
			'token_ids': padded_tokens
		}

if __name__ == '__main__':
	
	dataset = CocoDataset(
		annotations = Cfg.coco_train_ann,
		images_path = None,
		cache_path = None,
		dtype = DType.TEXT
	)
	
	collate_fn = CollateFn()

	train(
		dataset,
		collate_fn,
		output_dir = Cfg.upcap_output_dir,
		epochs = 50,
		start_epoch = 0
	)