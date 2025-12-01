import torch
import nltk
from torch import Tensor
from clip.simple_tokenizer import SimpleTokenizer

from yottacap.config import Cfg

class NLTKChunker:
	
	def __init__(self):
		self.tokenizer = SimpleTokenizer()
		grammar = r"""
			NP: {<DT|PP\$>?<JJ>*<NN.*>+}
		"""
		self.parser = nltk.RegexpParser(grammar)
		
		try:
			nltk.data.find('tokenizers/punkt')
			nltk.data.find('taggers/averaged_perceptron_tagger')
		except LookupError:
			nltk.download('punkt')
			nltk.download('averaged_perceptron_tagger')
		
	def get_chunk_ids(self, token_ids: Tensor) -> Tensor:
		
		L = token_ids.shape[0]
		chunk_ends = torch.arange(L, device=Cfg.device)
		words = []
		valid_ids = []
		
		for i, tid in enumerate(token_ids):
			t_item = tid.item()
			if t_item == 0 or t_item == Cfg.eos_token_id or t_item == Cfg.sot_token_id:
				continue
			text = self.tokenizer.decode([t_item]).strip()
			if text:
				words.append(text)
				valid_ids.append(i)
		
		if not words:
			return chunk_ends
		
		tagged = nltk.pos_tag(words)
		tree = self.parser.parse(tagged)
		
		current_word_idx = 0
		
		for subtree in tree:
			if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
				phrase_len = len(subtree.leaves())
				if current_word_idx + phrase_len - 1 < len(valid_ids):
					start_map = valid_ids[current_word_idx]
					end_map = valid_ids[current_word_idx + phrase_len - 1]
					chunk_ends[start_map : end_map + 1] = end_map
				current_word_idx += phrase_len
			else:
				current_word_idx += 1
		
		return chunk_ends