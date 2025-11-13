from torch.utils.data import Dataset
from pathlib import Path

from clipvl.layer.clipvl import MappingType
from clipvl.engine.train_text_image import train_text_image

def train(
	dataset: Dataset,
	output_dir: Path,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
	mapping_type: MappingType = MappingType.MLP,
	prefix_only: bool = True
):
	
	return train_text_image(
		dataset,
		output_dir,
		epochs,
		start_epoch,
		init_weights,
		mapping_type,
		prefix_only
	)
