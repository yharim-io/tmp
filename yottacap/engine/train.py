import torch
import torch.distributed as dist
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
):
	local_rank = int(os.environ["LOCAL_RANK"])
	device = torch.device(f"cuda:{local_rank}")
	torch.cuda.set_device(device)
	dist.init_process_group(backend='nccl')
	
	is_master = (dist.get_rank() == 0)
	
	model = YottaCap().to(device)
	model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
	
	# Main optimizer for Adapters + GPT + Projections
	# Discriminator is handled separately
	main_params = [p for n, p in model.named_parameters() if 'discriminator' not in n]
	optimizer = AdamW(main_params, lr=Cfg.learning_rate)
	
	disc_params = model.module.discriminator.parameters()
	disc_optimizer = AdamW(disc_params, lr=Cfg.discriminator_lr)
	
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=Cfg.batch_size, shuffle=True
	)
	
	for epoch in range(start_epoch, epochs + start_epoch):
		if epoch < Cfg.warmup_epochs:
			if is_master: print(f"--- Epoch {epoch}: Warmup ---")
			train_warmup(dataloader, model.module, optimizer)
		else:
			if is_master: print(f"--- Epoch {epoch}: Adversarial ---")
			train_adversarial(dataloader, model, optimizer, disc_optimizer)
			
		if is_master:
			if not output_dir.exists(): output_dir.mkdir(parents=True)
			torch.save(model.module.state_dict(), output_dir / f"epoch_{epoch}.pt")
			
	dist.destroy_process_group()