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

	def __call__(self, batch_data):
		raw_texts = [item['text'] for item in batch_data]
		
		batch_concepts_list: list[Tensor] = []
		batch_captions_list: list[Tensor] = []
		
		for text in raw_texts:
			concept_strings = self.parser(text)
			
			# [N_concepts, 77]
			sample_concepts_tensor: Tensor = clip.tokenize(concept_strings, truncate=True)
			batch_concepts_list.append(sample_concepts_tensor)
			
			# [77]
			# 优化：TextParser 的第一个元素就是全文，直接复用即可，无需再次 tokenize
			sample_caption_tensor: Tensor = sample_concepts_tensor[0]
			batch_captions_list.append(sample_caption_tensor)

		batch_size = len(batch_data)
		
		# 预分配 Tensor (Padding 默认为 0)
		padded_concept_batch: Tensor = torch.zeros(batch_size, Cfg.max_concepts, 77, dtype=torch.long)
		padded_caption_batch: Tensor = torch.zeros(batch_size, Cfg.max_seq_length, dtype=torch.long)

		for i, (sample_concepts, sample_caption) in enumerate(zip(batch_concepts_list, batch_captions_list)):
			
			# 处理概念输入 (截断 + 填充)
			num_concepts = min(len(sample_concepts), Cfg.max_concepts)
			padded_concept_batch[i, :num_concepts] = sample_concepts[:num_concepts]
			
			# 处理标题目标 (截断 + 填充)
			num_tokens = min(len(sample_caption), Cfg.max_seq_length)
			padded_caption_batch[i, :num_tokens] = sample_caption[:num_tokens]

		return {
			'text_concept_tokens': padded_concept_batch,
			'token_ids': padded_caption_batch
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