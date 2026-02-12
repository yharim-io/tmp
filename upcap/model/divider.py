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

	def process(
		self,
		image_path: os.PathLike,
		bg: bool = True,
		hidden_size: int = 320,
		flatten: bool = True,
	) -> Tensor:
		return self.process_batch([image_path], bg, hidden_size, flatten)

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
		flatten: bool = True,
		image_rgbs: list[np.ndarray] | None = None,
		output_size: int = 640,
	) -> Tensor | list[Tensor]:
		
		batch_tensors = []
		imgs = image_rgbs
		if imgs is None:
			imgs = []
			for p in image_paths:
				img = cv2.imread(str(p))
				if img is None:
					continue
				imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		for img in imgs:
			
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
		
		proc_batch = tensor_batch * 255.0
		processed_results = [
			self._process_single_result(proc_batch[i], res, bg, output_size)
			for i, res in enumerate(results)
		]

		if flatten:
			output_tensors = [out for out in processed_results if out is not None]

			if not output_tensors:
				return torch.empty(0)
			
			return torch.cat(output_tensors, dim=0)
		else:
			return [
				out if out is not None else torch.empty(0, device=Cfg.device)
				for out in processed_results
			]

	def _process_single_result(self, image: Tensor, result: Results, bg: bool, output_size: int) -> Tensor | None:
		if result.masks is None:
			return None

		# result.masks.data 已经在 GPU 上
		masks = result.masks.data
		# 移除按类别合并逻辑，直接使用原始的所有实例掩码
		if masks.shape[0] == 0:
			return None
		
		# 在 CPU 上做 NMS 选择，避免循环中频繁 GPU 标量同步
		merged_masks_cpu = masks.detach().to('cpu')
		areas_cpu = merged_masks_cpu.sum(dim=(1, 2))
		sorted_idx_cpu = torch.argsort(areas_cpu, descending=True)
		keep_indices_cpu: list[int] = []
		occupied_cpu = torch.zeros_like(merged_masks_cpu[0], dtype=torch.bool)

		for idx_t in sorted_idx_cpu:
			idx = int(idx_t.item())
			area = float(areas_cpu[idx].item())
			if area < 1000:
				continue

			curr = merged_masks_cpu[idx].bool()
			overlap = float((curr & occupied_cpu).sum().item()) / area
			if overlap > 0.1:
				continue

			keep_indices_cpu.append(idx)
			occupied_cpu |= curr
			
		if not keep_indices_cpu:
			return None

		keep_indices = torch.tensor(keep_indices_cpu, device=masks.device, dtype=torch.long)
		final_masks = masks[keep_indices].unsqueeze(-1) # (K, H, W, 1)
		occupied = occupied_cpu.to(masks.device)
		
		# 转换图像格式 (3, H, W) -> (H, W, 3) 以匹配后续操作
		image_hwc = image.permute(1, 2, 0)
		
		masked_imgs = image_hwc.unsqueeze(0) * final_masks
		
		if bg:
			fill_bg_mask = self.dilate_mask(occupied)
			filled_bg = self.inpainter(
				image_hwc.cpu(),
				fill_bg_mask.cpu()
			).to(image_hwc.device)
			masked_imgs = torch.cat([masked_imgs, filled_bg.unsqueeze(0)], dim=0)
		
		# Resize
		masked_imgs = masked_imgs.permute(0, 3, 1, 2)
		masked_imgs = F.interpolate(masked_imgs, size=(output_size, output_size), mode='bilinear', align_corners=False)
		masked_imgs = masked_imgs.permute(0, 2, 3, 1)
		
		return masked_imgs
	
if __name__ == "__main__":

	divider = Divider()
	
	with timer('Divider', 'process bacth'):
		results = divider.process_batch([
			Cfg.root/'data/example/1.jpg',
			Cfg.root/'data/example/2.jpg',
			Cfg.root/'data/example/3.jpg',
			Cfg.root/'data/example/4.jpg',
			Cfg.root/'data/example/5.jpg',
			Cfg.root/'data/example/6.jpg',
			Cfg.root/'data/example/7.jpg'
		])

	for i, img_tensor in enumerate(results):
		img = img_tensor.cpu().numpy().astype(np.uint8)
		output_path = Cfg.root / f"data/example/result/extracted_batch-{i}.png"
		cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))