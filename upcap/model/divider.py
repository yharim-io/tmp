import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.engine.results import Results
import os

from .inpainter import Inpainter
from utils.logger import timer
from upcap.config import Cfg

class Divider:

	def __init__(self):
		self.model = YOLO('yolo11n-seg.pt')
		self.inpainter = Inpainter()
		self.warmup()

	def warmup(self):
		tmp = Cfg.root / 'data/example/warmup_pixel.jpg'
		os.makedirs(tmp.parent, exist_ok=True)
		cv2.imwrite(str(tmp), np.zeros((640, 640, 3), dtype=np.uint8))
		try:
			self.process(tmp)
		finally:
			if tmp.exists():
				os.remove(tmp)

	def process(self, image_path: os.PathLike, bg: bool = True, hidden_size: int = 320) -> Tensor:
		return self.process_batch([image_path], bg, hidden_size)

	def dilate_mask(self, mask: Tensor, kernel_size: int = 15) -> Tensor:
		mask_float = mask.float().unsqueeze(0).unsqueeze(0)
		padding = kernel_size // 2
		dilated = F.max_pool2d(mask_float, kernel_size, stride=1, padding=padding)
		return dilated.squeeze(0).squeeze(0) > 0.5

	def process_batch(
		self,
		image_paths: list[os.PathLike],
		bg: bool = True,
		hidden_size: int = 320,
		flatten: bool = True
	) -> Tensor | list[Tensor]:
		
		batch_tensors = []
		
		for p in image_paths:
			img = cv2.imread(str(p))
			if img is None:
				continue
				
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			
			t = torch.from_numpy(img).to(Cfg.device)
			t = t.permute(2, 0, 1).float().unsqueeze(0) / 255.0
			
			t = F.interpolate(t, size=(hidden_size, hidden_size), mode='bilinear', align_corners=False)
			batch_tensors.append(t)
		
		if not batch_tensors:
			return torch.empty(0)

		tensor_batch = torch.cat(batch_tensors, dim=0)

		results: list[Results] = self.model(
			tensor_batch, 
			device=Cfg.device,
			retina_masks=False,
			imgsz=hidden_size,
			conf=0.2, 
			iou=0.9,
			half=True,
			verbose=False,
			stream=False
		)
		
		if flatten:
			output_tensors = []
			proc_batch = tensor_batch * 255.0

			for i, res in enumerate(results):
				out = self._process_single_result(proc_batch[i], res, bg)
				if out is not None:
					output_tensors.append(out)

			if not output_tensors:
				return torch.empty(0)
			
			return torch.cat(output_tensors, dim=0)
		else:
			output_list = []
			proc_batch = tensor_batch * 255.0

			for i, res in enumerate(results):
				out = self._process_single_result(proc_batch[i], res, bg)
				output_list.append(out if out is not None else torch.empty(0, device=Cfg.device))

			return output_list

	def _process_single_result(self, image: Tensor, result: Results, bg: bool) -> Tensor | None:
		if result.masks is None:
			return None

		# result.masks.data 已经在 GPU 上
		masks = result.masks.data
		cls_ids = result.boxes.cls.int()

		# GPU 上的掩码合并逻辑
		unique_cls = torch.unique(cls_ids)
		merged_list = []
		for c in unique_cls:
			merged_list.append(masks[cls_ids == c].amax(dim=0))
		
		if not merged_list:
			return None
			
		merged_masks = torch.stack(merged_list)
		areas = merged_masks.sum(dim=(1, 2))
		
		sorted_idx = torch.argsort(areas, descending=True)
		keep_indices = []
		occupied = torch.zeros_like(merged_masks[0], dtype=torch.bool)
		
		# NMS 逻辑
		for idx in sorted_idx:
			if areas[idx] < 1000:
				continue
			
			curr = merged_masks[idx].bool()
			# 使用 Tensor 操作计算重叠率
			overlap = (curr & occupied).sum() / areas[idx]
			
			if overlap > 0.1:
				continue
				
			keep_indices.append(idx)
			occupied = occupied | curr
			
		if not keep_indices:
			return None

		final_masks = merged_masks[torch.stack(keep_indices)].unsqueeze(-1) # (K, H, W, 1)
		
		# 转换图像格式 (3, H, W) -> (H, W, 3) 以匹配后续操作
		image_hwc = image.permute(1, 2, 0)
		
		masked_imgs = image_hwc.unsqueeze(0) * final_masks
		
		if bg:
			fill_bg_mask = self.dilate_mask(occupied)
			filled_bg = self.inpainter(image_hwc, fill_bg_mask)
			masked_imgs = torch.cat([masked_imgs, filled_bg.unsqueeze(0)], dim=0)
		
		# Resize to 640x640
		masked_imgs = masked_imgs.permute(0, 3, 1, 2)
		masked_imgs = F.interpolate(masked_imgs, size=(640, 640), mode='bilinear', align_corners=False)
		masked_imgs = masked_imgs.permute(0, 2, 3, 1)
		
		return masked_imgs
	
if __name__ == "__main__":

	divider = Divider()
	
	with timer('Divider', 'process bacth'):
		results = divider.process_batch([
			Cfg.root/'data/example/1.jpg',
			Cfg.root/'data/example/2.jpg',
			Cfg.root/'data/example/3.jpg'
		])

	for i, img_tensor in enumerate(results):
		img = img_tensor.cpu().numpy().astype(np.uint8)
		output_path = Cfg.root / f"data/example/result/extracted_batch-{i}.png"
		cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))