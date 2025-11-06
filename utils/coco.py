import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import clip
from clip.model import CLIP
from torchvision.transforms import Compose
import json
from pathlib import Path
import pickle
from PIL import Image
from tqdm import tqdm
from enum import IntFlag
import os

from utils.config import Config

class DataType(IntFlag):
	TEXT      = 1
	IMAGE     = 2
	TEXT_EMB  = 4
	IMAGE_EMB = 8

class _ImagePreloadDataset(Dataset):
	
	def __init__(self, images_paths: list[str], preprocess: Compose):
		self.images_paths = images_paths
		self.preprocess = preprocess
	
	def __len__(self) -> int:
		return len(self.images_paths)
	
	def __getitem__(self, index):
		image_path = self.images_paths[index]
		image = Image.open(image_path).convert('RGB')
		return self.preprocess(image)

class CocoDataset(Dataset):

	def __init__(
		self,
		annotation: Path,
		image_path: Path | None = None,
		cache_path: Path | None = None,
		data_type: DataType = 0,
		clip_model: CLIP | None = None,
		preprocess: Compose | None = None,
	):
		
		if DataType.IMAGE in data_type or DataType.IMAGE_EMB in data_type:
			assert image_path is not None, 'image data requires image_path'
		
		self.data_type: DataType = data_type
		
		self.texts: list[str] | None = None
		self.texts_embs: Tensor | None = None
		self.images: list[str] | None = None
		self.images_embs: Tensor | None = None
		
		if cache_path is not None and cache_path.exists():
			with open(cache_path, 'rb') as f:
				cache_data: dict = pickle.load(f)
				self.texts = cache_data.get('texts')
				self.images = cache_data.get('images')
				self.texts_embs = cache_data.get('texts_embs')
				self.images_embs = cache_data.get('images_embs')
			
			if ((DataType.TEXT in data_type and self.texts is None) or
				(DataType.IMAGE in data_type and self.images is None) or
				(DataType.TEXT_EMB in data_type and self.texts_embs is None) or
				(DataType.IMAGE_EMB in data_type and self.images_embs is None)
			):
				print("Cache miss or incomplete. Rebuilding...")
				self.texts = None
				self.texts_embs = None
				self.images = None
				self.images_embs = None
			else:
				return
		
		with open(annotation, 'r') as f:
			data = json.load(f)
		
		if DataType.TEXT in data_type or DataType.TEXT_EMB in data_type:
			self.texts = []
		
		if DataType.IMAGE in data_type or DataType.IMAGE_EMB in data_type:
			self.images = []
			image_id_to_filename: dict = {
				img['id']: img['file_name']
				for img in data['images']
			}
		
		for ann in data['annotations']:
			if DataType.TEXT in data_type or DataType.TEXT_EMB in data_type:
				self.texts.append(ann['caption'])
			if DataType.IMAGE in data_type or DataType.IMAGE_EMB in data_type:
				image_id = ann['image_id']
				filename = image_id_to_filename[image_id]
				self.images.append(str(image_path / filename))
		
		if DataType.IMAGE_EMB in data_type:
			
			if clip_model is None or preprocess is None:
				clip_model, preprocess = clip.load(
					Config.clip_pretrained_path,
					jit=False
				)
			
			self.images_embs = self._compute_image_embeddings(self.images, clip_model, preprocess)
			
			if not DataType.IMAGE in data_type:
				self.images = None
		
		if DataType.TEXT_EMB in data_type:
			
			self.texts_embs = self._compute_text_embeddings(self.texts)
			
			if not DataType.TEXT in data_type:
				self.texts = None
		
		if cache_path:
			cache_data: dict = {
				'texts': self.texts,
				'images': self.images,
				'texts_embs': self.texts_embs,
				'images_embs': self.images_embs
			}
			
			with open(cache_path, 'wb') as f:
				pickle.dump(cache_data, f)
	
	@torch.no_grad()
	def _compute_text_embeddings(
		self,
		texts: list[str],
		batch_size: int = 512
	) -> Tensor:
		
		n_texts: int = len(texts)
		n_batches = (n_texts + batch_size - 1) // batch_size
		
		temp_token_batches: list[Tensor] = []
		
		for i in tqdm(range(n_batches), desc='Encoding texts'):
			start_idx = i * batch_size
			end_idx = min((i + 1) * batch_size, n_texts)
			text_batch = texts[start_idx:end_idx]
			token_batch = clip.tokenize(text_batch).long()
			temp_token_batches.append(token_batch.cpu())
		
		return torch.cat(temp_token_batches, dim=0)
	
	@torch.no_grad()
	def _compute_image_embeddings(
		self,
		images_paths: list[str],
		clip_model: CLIP,
		preprocess: Compose,
		batch_size: int = 128
	) -> Tensor:
		
		n_images: int = len(images_paths)
		emb_dim: int = clip_model.visual.output_dim
		
		images_embs: Tensor = torch.empty(
			(n_images, emb_dim),
			dtype=torch.float32,
			device='cpu'
		)
		
		clip_model.eval()
		
		helper_dataset = _ImagePreloadDataset(images_paths, preprocess)
		
		num_workers = int(os.environ.get('OMP_NUM_THREADS', 8))
		
		data_loader = DataLoader(
			helper_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=True
		)
		
		current_index: int = 0
		
		for batch_tensor_cpu in tqdm(data_loader, desc='Encoding images'):
			batch_tensor_gpu = batch_tensor_cpu.to(Config.device, non_blocking=True)
			emb_batch_gpu: Tensor = clip_model.encode_image(batch_tensor_gpu)
			batch_len = emb_batch_gpu.shape[0]
			images_embs[current_index : current_index + batch_len] = emb_batch_gpu.cpu()
			current_index += batch_len
		
		return images_embs
		
		# for i, image_path in enumerate(tqdm(images_paths, desc='Encoding images')):
		# 	if Path(image_path).exists():
		# 		image = Image.open(image_path).convert('RGB')
		# 		image = preprocess(image).unsqueeze(0).to(Config.device)
		# 		emb = clip_model.encode_image(image)
		# 		emb = emb.cpu()
		# 	else:
		# 		emb = torch.zeros(1, emb_dim)
		# 	images_embs[i] = emb
		
		# return images_embs
	
	def __len__(self) -> int:
		if DataType.TEXT in self.data_type:
			return len(self.texts)
		if DataType.IMAGE in self.data_type:
			return len(self.images)
		if DataType.TEXT_EMB in self.data_type:
			return self.texts_embs.shape[0]
		if DataType.IMAGE_EMB in self.data_type:
			return self.images_embs.shape[0]
		return 0
	
	def __getitem__(self, index: int) -> dict[str, any]:
		item: dict = {}
		if DataType.TEXT in self.data_type:
			item['text'] = self.texts[index]
		if DataType.IMAGE in self.data_type:
			item['image'] = self.images[index]
		if DataType.TEXT_EMB in self.data_type:
			item['text_emb'] = self.texts_embs[index]
		if DataType.IMAGE_EMB in self.data_type:
			item['image_emb'] = self.images_embs[index]
		return item
