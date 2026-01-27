import torch
import torch.distributed as dist
from datetime import timedelta
from contextlib import contextmanager

from utils.config import Config as Cfg

@contextmanager
def dist_startup(
	device: torch.device | None = None,
	seed: int = 42,
	timeout: timedelta = timedelta(minutes=114514)
):
	dist.init_process_group(
		backend='nccl',
		init_method='env://',
		timeout=timeout,
	)
	if device is not None:
		torch.cuda.set_device(device)
	else:
		torch.cuda.set_device(Cfg.device)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	try:
		yield
	except:
		raise
	finally:
		dist.destroy_process_group()