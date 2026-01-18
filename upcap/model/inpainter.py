import cv2
import numpy as np
import torch
from torch import Tensor

class Inpainter:
	def __init__(self):
		pass

	def __call__(self, image: Tensor, mask: Tensor, size: int = 64) -> Tensor:
		img_np = image.cpu().numpy().astype(np.uint8)
		msk_np = mask.cpu().numpy().astype(np.uint8) * 255
		
		h, w = img_np.shape[:2]
		
		img_small = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_AREA)
		msk_small = cv2.resize(msk_np, (size, size), interpolation=cv2.INTER_NEAREST)
		
		out_small = cv2.inpaint(img_small, msk_small, 3, cv2.INPAINT_TELEA)
		out_np = cv2.resize(out_small, (w, h), interpolation=cv2.INTER_LINEAR)
		
		mask_bool = msk_np > 0
		if img_np.ndim == 3:
			mask_bool = mask_bool[..., None]
		
		out_np = np.where(mask_bool, out_np, img_np)
		
		return torch.from_numpy(out_np).to(image.device).float()