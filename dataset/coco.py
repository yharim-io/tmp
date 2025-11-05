from torch import Tensor
from torch.utils.data import Dataset
import clip
import json
import pandas as pd
from pathlib import Path
import pickle

class CocoDataset(Dataset):

	def __init__(
		self,
		train_data: Path | None = None,
		cache_path: Path | None = None,
		captions: list[str] | None = None,
	):
		
		self.captions: list[str] = []
		
		if cache_path is not None and cache_path.exists():
			with open(cache_path, 'rb') as f:
				cache_data = pickle.load(f)
				self.captions = cache_data['captions']
				self.token_ids_77 = cache_data['token_ids_77s']
			return
		
		elif captions is not None:
			self.captions = captions
			
		elif train_data is not None:
			with open(train_data, 'r') as f:
				data = json.load(f)
			df = pd.json_normalize(data['annotations'])
			self.captions = df['caption'].tolist()
			
		else:
			raise ValueError('one of train_data and captions is needed.')
		
		self.token_ids_77s = clip.tokenize(self.captions).long()
		
		if cache_path is not None:
			cache_path.parent.mkdir(parents=True, exist_ok=True)
			cache_data = {
				'captions': self.captions,
				'token_ids_77s': self.token_ids_77s
			}
			with open(cache_path, 'wb') as f:
				pickle.dump(cache_data, f)

	def __len__(self) -> int:
		return len(self.captions)
	
	def __getitem__(self, index: int) -> Tensor:
		return self.token_ids_77s[index]