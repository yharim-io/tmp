import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.engine.results import Results
from concurrent.futures import ThreadPoolExecutor
import os

from .inpainter import Inpainter
from utils.logger import timer
from upcap.config import Cfg

class Divider:

	def __init__(self):
		self.model = YOLO('yolo11n-seg.pt')
		self.inpainter = Inpainter()

	def process(self, image_path: os.PathLike) -> Tensor:
		image = cv2.imread(str(image_path))
		if image is None:
			return torch.empty(0)
		
		if (image.shape[1], image.shape[0]) != (640, 640):
			image = cv2.resize(image, (640, 640))
		
		results: Results = self.model(
			image, 
			device=Cfg.device,
			retina_masks=True,
			imgsz=640,
			conf=0.2, 
			iou=0.9,
			half=True,
			verbose=False
		)
		
		tensor = self._process_single_result(image, results[0])
		
		if tensor is None:
			return torch.empty(0)
			
		return tensor

	def dilate_mask(self, mask: Tensor, kernel_size: int = 15) -> Tensor:
		mask_float = mask.float().unsqueeze(0).unsqueeze(0)
		padding = kernel_size // 2
		dilated = F.max_pool2d(mask_float, kernel_size, stride=1, padding=padding)
		return dilated.squeeze(0).squeeze(0) > 0.5

	def process_batch(self, image_paths: list[os.PathLike]) -> Tensor:
		raw_images = []
		for p in image_paths:
			img = cv2.imread(str(p))
			if img is not None:
				if (img.shape[1], img.shape[0]) != (640, 640):
					img = cv2.resize(img, (640, 640))
				raw_images.append(img)
		
		if not raw_images:
			return torch.empty(0)

		results: list[Results] = self.model(
			raw_images, 
			device=Cfg.device,
			retina_masks=True,
			imgsz=640,
			conf=0.2, 
			iou=0.9,
			half=True,
			verbose=False,
			stream=False 
		)
		
		batch_tensors = [torch.empty((0, 640, 640, 3), device=Cfg.device) for _ in range(len(raw_images))]
		
		def run_single(idx, img, res):
			out = self._process_single_result(img, res)
			if out is not None:
				batch_tensors[idx] = out

		with ThreadPoolExecutor(max_workers=8) as executor:
			futures = [executor.submit(run_single, i, img, res) for i, (img, res) in enumerate(zip(raw_images, results))]
			for f in futures:
				f.result()
		
		return torch.cat(batch_tensors, dim=0)

	def _process_single_result(self, image: np.ndarray, result: Results) -> Tensor | None:
		if result.masks is None:
			return None

		masks = result.masks.data.cpu().numpy()

		cls_ids = result.boxes.cls.cpu().numpy().astype(int)
		merged_map = {}
		for mask_data, c_id in zip(masks, cls_ids):
			mask_bool = mask_data.astype(bool)
			if c_id not in merged_map:
				merged_map[c_id] = mask_bool
			else:
				merged_map[c_id] = np.logical_or(merged_map[c_id], mask_bool)
		
		if not merged_map:
			return None
			
		masks = np.stack(list(merged_map.values()))
		
		areas = np.sum(masks, axis=(1, 2))
		sorted_indices = np.argsort(areas)[::-1]
		
		keep_masks = []
		occupied_mask = np.zeros(masks.shape[1:], dtype=bool)

		for idx in sorted_indices:
			current_mask = masks[idx].astype(bool)
			current_area = areas[idx]
			
			if current_area < 1000:
				continue

			intersection = np.logical_and(current_mask, occupied_mask)
			overlap_ratio = np.sum(intersection) / current_area
			
			if overlap_ratio > 0.1:
				continue

			keep_masks.append(current_mask)
			occupied_mask = np.logical_or(occupied_mask, current_mask)

		if not keep_masks:
			return None

		masks_stack = np.stack(keep_masks)
		
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		masks_tensor = torch.from_numpy(masks_stack).to(Cfg.device).unsqueeze(-1)
		image_tensor = torch.from_numpy(image_rgb).to(Cfg.device).unsqueeze(0)

		masked_images = image_tensor * masks_tensor

		fill_bg_mask = self.dilate_mask(torch.from_numpy(occupied_mask).to(Cfg.device))
		filled_bg = self.inpainter(image_tensor[0], fill_bg_mask)

		masked_images = torch.cat([masked_images, filled_bg.unsqueeze(0)], dim=0)

		masked_images = masked_images.permute(0, 3, 1, 2).float()
		masked_images = F.interpolate(masked_images, size=(640, 640), mode='bilinear', align_corners=False)
		masked_images = masked_images.permute(0, 2, 3, 1)

		return masked_images

if __name__ == "__main__":

	divider = Divider()
	
	with timer('divider', 'dummy'):
		dummy = divider.process(Cfg.root/'data/example/1.jpg')
	
	results = divider.process_batch([
		Cfg.root/'data/example/1.jpg',
		Cfg.root/'data/example/2.jpg',
		Cfg.root/'data/example/3.jpg'
	])

	for i, img_tensor in enumerate(results):
		img = img_tensor.cpu().numpy().astype(np.uint8)
		output_path = Cfg.root / f"data/example/result/extracted_batch-{i}.png"
		cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))