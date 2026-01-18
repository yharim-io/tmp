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

	def __call__(self, batch):
		texts = [item['text'] for item in batch]
		
		all_concept_tokens = []
		all_tokens = []
		
		for text in texts:
			concepts = self.parser(text)
			
			concept_tokens = clip.tokenize(concepts, truncate=True) 
			all_concept_tokens.append(concept_tokens)
			
			tokens = clip.tokenize(text, truncate=True).squeeze(0)
			all_tokens.append(tokens)

		padded_concept_tokens = torch.zeros(len(batch), Cfg.max_concepts, 77, dtype=torch.long)
		padded_tokens = torch.zeros(len(batch), Cfg.max_seq_length, dtype=torch.long)

		for i, (c_tokens, t) in enumerate(zip(all_concept_tokens, all_tokens)):
			n_c = min(c_tokens.shape[0], Cfg.max_concepts)
			padded_concept_tokens[i, :n_c] = c_tokens[:n_c]
			
			n_t = min(t.shape[0], Cfg.max_seq_length)
			padded_tokens[i, :n_t] = t[:n_t]

		return {
			'text_concept_tokens': padded_concept_tokens,
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
		output_dir = Cfg.root/'data/upcap/coco',
		epochs = 50,
		start_epoch = 0
	)