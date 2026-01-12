import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from pathlib import Path
import os

from yottacap.layer.yottacap import YottaCap
from yottacap.config import Cfg
from yottacap.engine.train_warmup import train_warmup
from yottacap.engine.train_adversarial import train_adversarial

def train(
	dataset,
	output_dir: Path,
	epochs: int,
	start_epoch: int = 0,
	init_weights: Path | None = None,
):
	
	torch.cuda.set_device(Cfg.device)
	dist.init_process_group(backend='nccl', init_method='env://')
	torch.cuda.manual_seed_all(42)
	torch.manual_seed(42)
	
	yottacap_model = YottaCap()

	if init_weights is not None:
		yottacap_model.load_state_dict(
			torch.load(
				init_weights,
				map_location=torch.device('cpu'),
				weights_only=True
			)
		)

	yottacap_model.to(Cfg.device)
	yottacap_model = DDP(
		yottacap_model,
		device_ids=[Cfg.rank],
		output_device=Cfg.rank
	)
	
	main_params = [
		p for n, p in yottacap_model.named_parameters()
		if 'discriminator' not in n
	]
	text_optimizer = AdamW(main_params, lr=Cfg.learning_rate)
	img_optimizer = AdamW(main_params, lr=Cfg.learning_rate)

	disc_params = yottacap_model.module.discriminator.parameters()
	disc_optimizer = AdamW(disc_params, lr=Cfg.discriminator_learning_rate)
	
	dataloader = DataLoader(
		dataset, batch_size=Cfg.batch_size, shuffle=True
	)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		if epoch < Cfg.warmup_epochs:
			if Cfg.is_master: print(f"--- Epoch {epoch}: Warmup ---")
			train_warmup(dataloader, yottacap_model.module)
		else:
			if Cfg.is_master: print(f"--- Epoch {epoch}: Adversarial ---")
			train_adversarial(dataloader, yottacap_model.module, text_optimizer, img_optimizer, disc_optimizer)
			
		if Cfg.is_master:
			if not output_dir.exists(): output_dir.mkdir(parents=True)
			torch.save(yottacap_model.module.state_dict(), output_dir / f"epoch_{epoch}.pt")
			
	dist.destroy_process_group()