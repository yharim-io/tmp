import cv2
import numpy as np
import torch

class Inpainter:
    def __init__(self):
        pass

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        img_np = image.cpu().numpy()
        msk_np = mask.cpu().numpy().astype(np.uint8) * 255
        
        out_np = cv2.inpaint(img_np, msk_np, 3, cv2.INPAINT_TELEA)
        
        return torch.from_numpy(out_np).to(image.device)