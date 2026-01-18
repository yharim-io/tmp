from torch.utils.data import Dataset
from pathlib import Path

from decap.layer.decap import DeCap
from decap.engine.train_text_only import train_text_only
from decap.engine.train_text_image import train_text_image

def train(
	dataset: Dataset,
	output_dir: Path,
	epochs: int = 10,
	start_epoch: int = 0,
	init_weights: Path | None = None,
	text_only: bool = True
) -> DeCap:
	
	if text_only:
		return train_text_only(
			dataset,
			output_dir,
			epochs,
			start_epoch,
			init_weights
		)
	
	else:
		return train_text_image(
			dataset,
			output_dir,
			epochs,
			start_epoch,
			init_weights
		)