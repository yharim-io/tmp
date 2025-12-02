import torch
import nltk
import zipfile
from torch import Tensor
from clip.simple_tokenizer import SimpleTokenizer
from yottacap.config import Cfg
from utils.logger import logger

class NLTKChunker:
	
	def __init__(self):
		self.tokenizer = SimpleTokenizer()
		self.decoder = self.tokenizer.decoder 
		
		grammar = r"""
			NP: {<DT|PP\$>?<JJ>*<NN.*>+}
		"""
		self.parser = nltk.RegexpParser(grammar)
		
		with logger('NLTK', 'Checking data'):
			try:
				nltk.data.find('tokenizers/punkt')
				nltk.data.find('taggers/averaged_perceptron_tagger_eng')
			except (LookupError, zipfile.BadZipFile):
				nltk.download('punkt', quiet=False, force=True)
				nltk.download('averaged_perceptron_tagger_eng', quiet=False, force=True)
			except Exception:
				nltk.download('punkt', quiet=False, force=True)
				nltk.download('averaged_perceptron_tagger_eng', quiet=False, force=True)
		
	def get_chunk_ids(self, token_ids: Tensor) -> Tensor:
		
		L = token_ids.shape[0]
		chunk_ends = torch.arange(L, device=Cfg.device)
		
		words = []
		word_to_token_indices = []
		
		current_word_parts = []
		current_token_indices = []
		
		for i, tid in enumerate(token_ids):
			t_item = tid.item()
			
			if t_item in [0, Cfg.eos_token_id, Cfg.sot_token_id]:
				continue
			
			token_str = self.decoder.get(t_item, "")
			
			current_word_parts.append(token_str.replace('</w>', ''))
			current_token_indices.append(i)
			
			if token_str.endswith('</w>'):
				full_word = "".join(current_word_parts)
				words.append(full_word)
				word_to_token_indices.append(current_token_indices)
				
				current_word_parts = []
				current_token_indices = []

		if not words:
			return chunk_ends
		
		tagged = nltk.pos_tag(words)
		tree = self.parser.parse(tagged)
		
		word_idx = 0
		for subtree in tree:
			if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
				num_words_in_phrase = len(subtree.leaves())
				
				start_token_idx = word_to_token_indices[word_idx][0]
				end_token_idx = word_to_token_indices[word_idx + num_words_in_phrase - 1][-1]
				
				chunk_ends[start_token_idx : end_token_idx + 1] = end_token_idx
				
				word_idx += num_words_in_phrase
			else:
				word_idx += 1
				
		return chunk_ends

if __name__ == '__main__':
	
	chunker = NLTKChunker()
	
	sample_text = "The quick brown fox jumps over the lazy dog."
	token_ids = torch.tensor([Cfg.sot_token_id] + chunker.tokenizer.encode(sample_text) + [Cfg.eos_token_id])
	
	chunk_ids = chunker.get_chunk_ids(token_ids)
	
	print("Token IDs: ", token_ids.tolist())
	print("Chunk IDs: ", chunk_ids.tolist())