import clip
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from utils.config import Config

class MetricEvaluator:
	def __init__(
		self,
		clip_model: CLIP | None = None,
		preprocess: Compose | None = None,
		tokenizer: SimpleTokenizer | None = None
	):
		if clip_model is None or preprocess is None:
			clip_model, preprocess = clip.load(Config.clip_pretrained_path, device=Config.device, jit=False)
		if tokenizer is None:
			tokenizer = SimpleTokenizer()
		self.clip_model = clip_model
		self.preprocess = preprocess
		self.tokenizer = tokenizer

	def compute(
		self,
		ground_truths: list[str],
		predictions: list[str]
	) -> dict:
		
		scorers = [
			(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
			(Meteor(), "METEOR"),
			(Cider(), "CIDEr"),
			(Spice(), "SPICE")
		]
		
		metric_scores = {}
		for scorer, method_name in scorers:
			score, _ = scorer.compute_score(ground_truths, predictions)
			if isinstance(method_name, list):
				for m, s in zip(method_name, score):
					metric_scores[m] = s
			else:
				metric_scores[method_name] = score
		
		return metric_scores