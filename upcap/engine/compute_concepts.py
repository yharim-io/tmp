import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from clip.model import CLIP
from torchvision.transforms import Compose

from upcap.config import Cfg
from upcap.model.divider import Divider

@torch.no_grad()
def compute_concepts(
	dataset: Dataset,
	clip_model: CLIP,
	preprocess: Compose,
	divider: Divider
) -> Tensor:
	
	all_features: list[Tensor] = []
	
	for i in tqdm(range(len(dataset)), desc='Computing Concepts'):
		item = dataset[i]
		image_path = item.get('image')
		
		if not image_path:
			continue
		
		divided_images: Tensor = divider.process(image_path)
		
		if divided_images.numel() == 0:
			continue
		
		imgs_np = divided_images.cpu().numpy().astype(np.uint8)
		clip_inputs = []
		
		for img_arr in imgs_np:
			img_pil = Image.fromarray(img_arr)
			processed_img = preprocess(img_pil)
			clip_inputs.append(processed_img)
			
		clip_batch = torch.stack(clip_inputs).to(Cfg.device)
		
		features: Tensor = clip_model.encode_image(clip_batch)
		features /= features.norm(dim=-1, keepdim=True)
		
		all_features.append(features.cpu())

	if not all_features:
		return torch.empty(0)

	return torch.cat(all_features, dim=0)