import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import clip
from clip.model import CLIP
from torchvision.transforms import Compose
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from enum import IntFlag
import os

from utils.config import Config

class DType(IntFlag):
	TEXT       = 1
	TEXT_EMB   = 1 << 1
	TEXT_FEAT  = 1 << 2
	IMAGE      = 1 << 3
	IMAGE_EMB  = 1 << 4
	IMAGE_FEAT = 1 << 5
	
	HAS_TEXT   = TEXT | TEXT_EMB | TEXT_FEAT
	HAS_IMAGE  = IMAGE | IMAGE_EMB | IMAGE_FEAT
	NEED_CLIP  = TEXT_EMB | TEXT_FEAT | IMAGE_EMB | IMAGE_FEAT
	ALL        = HAS_TEXT | HAS_IMAGE
	NONE       = 0

TYPE_MAP: dict[DType, str] = {
	DType.TEXT: 'texts',
	DType.IMAGE: 'images',
	DType.TEXT_EMB: 'texts_embs',
	DType.TEXT_FEAT: 'texts_feats',
	DType.IMAGE_EMB: 'images_embs',
	DType.IMAGE_FEAT: 'images_feats',
}

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

class DatasetBase(Dataset):
	
	def __init__(
		self,
		cache_path: Path | None,
		dtype: DType,
		clip_model: CLIP | None = None,
		preprocess: Compose | None = None,
	):
		super().__init__()
		
		self.dtype: DType = dtype
		self.cache_path: Path | None = cache_path
		self._clip_model = clip_model
		self._preprocess = preprocess
		
		self.texts: list[str] | None = None
		self.images: list[str] | None = None
		self.texts_embs: Tensor | None = None
		self.texts_feats: Tensor | None = None
		self.images_embs: Tensor | None = None
		self.images_feats: Tensor | None = None
		
		self.gather_texts_images()
		self.validate_dependencies()
		
		remain_type: DType = self._load_cache()
		
		if remain_type != DType.NONE:
		
			self._init_clip()
			
			if DType.TEXT_EMB in remain_type:
				self._compute_texts_embs()
			if DType.TEXT_FEAT in remain_type:
				self._compute_texts_feats()
			if DType.IMAGE_EMB in remain_type:
				self._compute_images_embs()
			if DType.IMAGE_FEAT in remain_type:
				self._compute_images_feats()
			
			self._save_cache()
		
		for dt, attr in TYPE_MAP.items():
			if dt not in self.dtype:
				setattr(self, attr, None)
	
	def gather_texts_images(self):
		pass
	
	def validate_dependencies(self):
		if self.dtype & DType.HAS_TEXT:
			assert self.texts is not None, 'dtype requires text data'
		if self.dtype & DType.HAS_IMAGE:
			assert self.images is not None, 'dtype requires image data'
	
	def _init_clip(self):
		if self._clip_model is not None and self._preprocess is not None:
			return
		if not (self.dtype & DType.NEED_CLIP):
			return
		self._clip_model, self._preprocess = clip.load(
			Config.clip_pretrained_path,
			device=Config.device,
			jit=False
		)
	
	def _load_cache(self) -> DType:
		
		remain_type: DType = self.dtype
		
		if self.cache_path is None:
			return remain_type
		
		os.makedirs(self.cache_path, exist_ok=True)
		
		for dt, attr in TYPE_MAP.items():
			if dt in remain_type:
				pt_path = self.cache_path / f"{attr}.pt"
				if pt_path.exists():
					setattr(self, attr, torch.load(pt_path))
					remain_type ^= dt
		
		return remain_type
		
	def _save_cache(self):
		
		if self.cache_path is None:
			return
		
		os.makedirs(self.cache_path, exist_ok=True)
		
		for dt, attr in TYPE_MAP.items():
			data = getattr(self, attr)
			if data is not None:
				pt_path = self.cache_path / f"{attr}.pt"
				torch.save(data, pt_path)
	
	@torch.no_grad()
	def _compute_texts_embs(self, batch_size: int = 1024):
		n_texts = len(self.texts)
		n_batches = (n_texts + batch_size - 1) // batch_size
		texts_embs_list: list[Tensor] = []
		
		for i in tqdm(range(n_batches), desc='Computing texts_embs'):
			texts_batch = self.texts[i * batch_size : (i + 1) * batch_size]
			texts_embs_batch = clip.tokenize(texts_batch).long()
			texts_embs_list.append(texts_embs_batch)
		
		self.texts_embs = torch.cat(texts_embs_list, dim=0)
	
	@torch.no_grad()
	def _compute_texts_feats(self, batch_size: int = 1024):
		if self.texts_embs is None:
			self._compute_texts_embs()
		
		n_texts = len(self.texts)
		n_batches = (n_texts + batch_size - 1) // batch_size
		texts_feats_list: list[Tensor] = []
		
		for i in tqdm(range(n_batches), desc='Computing texts_feats'):
			texts_embs_batch = self.texts_embs[i * batch_size : (i + 1) * batch_size]
			texts_embs_batch = texts_embs_batch.to(Config.device)
			texts_feats_batch = self._clip_model.encode_text(texts_embs_batch)
			texts_feats_list.append(texts_feats_batch.cpu())
		
		self.texts_feats = torch.cat(texts_feats_list, dim=0)
	
	@torch.no_grad()
	def _compute_images_embs(self, batch_size: int = 128):
		
		dataset = _ImagePreloadDataset(self.images, self._preprocess)
		loader = DataLoader(
			dataset,
			batch_size=batch_size,
			num_workers=8,
			pin_memory=True
		)
		
		captured = { 'value': None }
		def hook_fn(module, input, output):
			captured['value'] = output
		
		handle = self._clip_model.visual.transformer.register_forward_hook(hook_fn)
		
		images_feats_list: list[Tensor] = []
		images_embs_list: list[Tensor] = []
		
		for images_batch in tqdm(loader, desc='Computing images_embs'):
			images_batch = images_batch.to(Config.device)
			images_feats_batch = self._clip_model.encode_image(images_batch)
			images_feats_list.append(images_feats_batch.cpu())
			images_embs_batch = captured['value'].permute(1, 0, 2)
			images_embs_batch = self._clip_model.visual.ln_post(images_embs_batch)
			images_embs_list.append(images_embs_batch.cpu())
		
		handle.remove()
		
		self.images_feats = torch.cat(images_feats_list, dim=0)
		self.images_embs = torch.cat(images_embs_list, dim=0)
		
	@torch.no_grad()
	def _compute_images_feats(self, batch_size: int = 128):
		
		if self.images_feats is not None:
			print('Computing images_feats: computed by _compute_images_embs')
			return
		
		dataset = _ImagePreloadDataset(self.images, self._preprocess)
		loader = DataLoader(
			dataset,
			batch_size=batch_size,
			num_workers=8,
			pin_memory=True
		)
		
		images_feats_list: list[Tensor] = []
		
		for images_batch in tqdm(loader, desc='Computing images_feats'):
			images_batch = images_batch.to(Config.device)
			images_feats_batch = self._clip_model.encode_image(images_batch)
			images_feats_list.append(images_feats_batch.cpu())
		
		self.images_feats = torch.cat(images_feats_list, dim=0)
	
	def __len__(self) -> int:
		if self.texts is not None:
			return len(self.texts)
		if self.images is not None:
			return len(self.images)
		if self.texts_embs is not None:
			return self.texts_embs.shape[0]
		if self.texts_feats is not None:
			return self.texts_feats.shape[0]
		if self.images_embs is not None:
			return self.images_embs.shape[0]
		if self.images_feats is not None:
			return self.images_feats.shape[0]
		return 0
	
	def __getitem__(self, index) -> dict[str, list[str] | Tensor]:
		item: dict = {}
		if DType.TEXT in self.dtype:
			item['text'] = self.texts[index]
		if DType.TEXT_EMB in self.dtype:
			item['text_emb'] = self.texts_embs[index]
		if DType.TEXT_FEAT in self.dtype:
			item['text_feat'] = self.texts_feats[index]
		if DType.IMAGE in self.dtype:
			item['image'] = self.images[index]
		if DType.IMAGE_EMB in self.dtype:
			item['image_emb'] = self.images_embs[index]
		if DType.IMAGE_FEAT in self.dtype:
			item['image_feat'] = self.images_feats[index]
		return item

class CocoDataset(DatasetBase):

	def __init__(
		self,
		annotations: Path,
		images_path: Path | None,
		cache_path: Path | None,
		dtype: DType,
		clip_model: CLIP | None = None,
		preprocess: Compose | None = None,
	):
		self.annotations = annotations
		self.images_path = images_path
		super().__init__(
			cache_path,
			dtype,
			clip_model,
			preprocess
		)
	
	def gather_texts_images(self):
		
		with open(self.annotations, 'r') as f:
			data = json.load(f)
		
		has_text: bool = bool(self.dtype & DType.HAS_TEXT)
		has_image: bool = bool(self.dtype & DType.HAS_IMAGE)
		
		if has_text:
			self.texts = []
		
		if has_image:
			self.images = []
			image_id_to_filename = {
				img['id']: img['file_name']
				for img in data['images']
			}
		
		for ann in data['annotations']:
			if has_text:
				self.texts.append(ann['caption'])
			if has_image:
				image_id = ann['image_id']
				filename = image_id_to_filename[image_id]
				self.images.append(str(self.images_path / filename))
		
		return super().gather_texts_images()

if __name__ == '__main__':
	
	# dataset = CocoDataset(
	# 	annotations=Config.coco_train_ann,
	# 	images_path=Config.coco_train_image,
	# 	cache_path=Config.coco_train_cache,
	# 	dtype=DType.ALL
	# )
	
	dataset = CocoDataset(
		annotations=Config.coco_val_ann,
		images_path=Config.coco_val_image,
		cache_path=Config.coco_val_cache,
		dtype=DType.ALL
	)
