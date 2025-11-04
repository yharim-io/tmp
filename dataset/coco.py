from torch.utils.data import Dataset
import json
import pandas as pd
from pathlib import Path

class CocoDataset(Dataset):

	def __init__(
		self,
		train_data: Path | None = None,
		captions: list[str] | None = None
	):
		
		if captions is not None:
			self.captions = captions
			return
		
		assert train_data is not None, \
			"one of train_data and captions needed."
		
		self.captions: list[str] = []
		
		with open(train_data, 'r') as f:
			data = json.load(f)
			df = pd.json_normalize(data['annotations'])
			self.captions = df['caption'].tolist()

	def __len__(self) -> int:
		return len(self.captions)
	
	def __getitem__(self, index: int) -> str:
		return self.captions[index]