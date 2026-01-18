import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from upcap.config import Cfg
from upcap.model.divider import Divider

@torch.inference_mode()
def compute_concepts(
    dataset: Dataset,
    divider: Divider,
    batch_size: int = 64
) -> Tensor:
    
    sampler = DistributedSampler(dataset, shuffle=False)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=lambda x: x, 
        pin_memory=True
    )
    
    all_images: list[Tensor] = []
    
	# https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=Cfg.device).view(1, 3, 1, 1).half()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=Cfg.device).view(1, 3, 1, 1).half()
    
    if Cfg.is_master:
        iterator = tqdm(dataloader, desc='Extracting Concept Images')
    else:
        iterator = dataloader
    
    for batch in iterator:
        image_paths = [item.get('image') for item in batch if item.get('image')]
        if not image_paths:
            continue

        divided_images: Tensor = divider.process_batch(image_paths)
        
        if divided_images.numel() == 0:
            continue
        
        x = divided_images.permute(0, 3, 1, 2).half()
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = x / 255.0
        x = (x - mean) / std
        
        all_images.append(x.cpu())

    if not all_images:
        return torch.empty(0)

    return torch.cat(all_images, dim=0)