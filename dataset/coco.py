from torch.utils.data import Dataset
import json
import random
import pandas as pd
from pathlib import Path

class CocoDataset(Dataset):

	def __init__(self, train_data: Path):
		
		self.captions: list[str] = []
		
		with open(train_data, 'r') as f:
			data = json.load(f)
			df = pd.json_normalize(data['annotations'])
			self.captions = df['caption'].tolist()
			
		self.shuffle()

	def shuffle(self):
		random.shuffle(self.captions)

	def __len__(self) -> int:
		return len(self.captions)
	
	def __getitem__(self, index: int) -> str:
		return self.captions[index]