import numpy as np
import torch
from torch import nn
import clip
from clip.simple_tokenizer import SimpleTokenizer
import numpy as np

from zerocap.layer.gpt2 import GPT2
from zerocap.config import Cfg

def add_context(x, y):
	return (x[0] + y[0], x[1] + y[1])

class ZeroCap(nn.Module):
	
	def __init__(self, clip_model: clip.model.CLIP):
		super().__init__()
		self.device = Cfg.device
		torch.manual_seed(42)
		np.random.seed(42)

		self.gpt = GPT2().to(self.device)
		self.gpt.eval()
		for param in self.gpt.parameters():
			param.requires_grad = False
		
		self.tokenizer = SimpleTokenizer()
		self.context_prefix_id = Cfg.sot_token_id
		self.context_prefix_str = self.tokenizer.decode([self.context_prefix_id])
		
		self.forbidden_tokens = np.load(Cfg.forbidden_tokens_path)
		self.end_token_id = self.tokenizer.encode(Cfg.end_token)[0]

		self.clip_model = clip_model
		self.image_features = None

	def get_txt_features(self, text: list[str]) -> torch.Tensor:
		clip_texts = clip.tokenize(text).to(self.device)
		with torch.no_grad():
			text_features = self.clip_model.encode_text(clip_texts).float()
			text_features = text_features / text_features.norm(dim=-1, keepdim=True)
		return text_features.detach()

	def clip_loss(self, probs: torch.Tensor, context_tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
		top_size = 512
		_, top_indices = probs.topk(top_size, -1)
		
		prefix_texts = [
			self.tokenizer.decode(x.cpu().numpy().tolist()).replace(self.context_prefix_str, '')
			for x in context_tokens
		]

		clip_loss = 0
		losses = []
		for i in range(probs.shape[0]):
			top_texts = []
			prefix_text = prefix_texts[i]
			for x in top_indices[i]:
				top_texts.append(prefix_text + self.tokenizer.decode([x.item()]))
			
			text_features = self.get_txt_features(top_texts)

			with torch.no_grad():
				sim = (self.image_features[i:(i+1)] @ text_features.T)
				target_probs = nn.functional.softmax(sim / Cfg.clip_loss_temperature, dim=-1).detach()
				target_probs = target_probs.type(torch.float32)

			target = torch.zeros_like(probs[i])
			target[top_indices[i]] = target_probs[0]
			target = target.unsqueeze(0)
			cur_clip_loss = torch.sum(-(target * torch.log(probs[i:(i + 1)])))

			clip_loss += cur_clip_loss
			losses.append(cur_clip_loss)

		return clip_loss, losses

	def update_logits(self, context_tokens: torch.Tensor, i: int, logits: torch.Tensor) -> torch.Tensor:
		for beam_id in range(context_tokens.shape[0]):
			for token_idx in set(context_tokens[beam_id][-4:].tolist()):
				factor = Cfg.repetition_penalty if logits[beam_id, token_idx] > 0 else (1 / Cfg.repetition_penalty)
				logits[beam_id, token_idx] /= factor

			if i >= 1:
				factor = Cfg.end_factor if logits[beam_id, self.end_token_id] > 0 else (1 / Cfg.end_factor)
				logits[beam_id, self.end_token_id] *= factor
			
			for token_idx in list(self.forbidden_tokens):
				factor = Cfg.forbidden_factor if logits[beam_id, token_idx] > 0 else (1 / Cfg.forbidden_factor)
				logits[beam_id, token_idx] /= factor
		
		logits[:, Cfg.clip_vocab_size:] = -torch.inf
		return logits

	def shift_context(self, context: tuple, last_token: torch.Tensor, context_tokens: torch.Tensor, probs_before_shift: torch.Tensor) -> tuple:
		context_delta = [tuple([np.zeros(x.shape).astype("float32") for x in p]) for p in context]

		for _ in range(Cfg.num_iterations):
			curr_shift = [
				tuple([torch.tensor(x, requires_grad=True, device=self.device) for x in p_])
				for p_ in context_delta
			]
			
			for p0, p1 in curr_shift:
				p0.retain_grad()
				p1.retain_grad()

			shifted_context = list(map(add_context, context, curr_shift))
			shifted_outputs = self.gpt.core(last_token, past_key_values=shifted_context)
			logits = shifted_outputs["logits"][:, -1, :]
			logits[:, Cfg.clip_vocab_size:] = -torch.inf
			probs = nn.functional.softmax(logits, dim=-1)

			loss = 0.0
			clip_loss, _ = self.clip_loss(probs, context_tokens)
			loss += Cfg.clip_scale * clip_loss

			ce_loss = Cfg.ce_scale * ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
			loss += ce_loss.sum()
			loss.backward()

			sep_grads = None
			for b in range(context_tokens.shape[0]):
				tmp_sep_norms = [[(torch.norm(x.grad[b:(b + 1)]) + 1e-15) for x in p_] for p_ in curr_shift]
				tmp_grad = [
					tuple([-Cfg.stepsize * (x.grad[b:(b + 1)] / tmp_sep_norms[i][j] ** Cfg.grad_norm_factor).data.cpu().numpy()
						   for j, x in enumerate(p_)])
					for i, p_ in enumerate(curr_shift)
				]
				if sep_grads is None:
					sep_grads = tmp_grad
				else:
					for l_index in range(len(sep_grads)):
						sep_grads[l_index] = list(sep_grads[l_index])
						for k_index in range(len(sep_grads[0])):
							sep_grads[l_index][k_index] = np.concatenate((sep_grads[l_index][k_index], tmp_grad[l_index][k_index]), axis=0)
						sep_grads[l_index] = tuple(sep_grads[l_index])
			
			context_delta = list(map(add_context, sep_grads, context_delta))
			
			for p0, p1 in curr_shift:
				p0.grad.data.zero_()
				p1.grad.data.zero_()
			
			context = [(p0.detach(), p1.detach()) for p0, p1 in context]

		context_delta = [
			tuple([torch.tensor(x, requires_grad=True, device=self.device) for x in p_])
			for p_ in context_delta
		]
		context = list(map(add_context, context, context_delta))
		context = [(p0.detach(), p1.detach()) for p0, p1 in context]
		return context

	def get_next_probs(self, i: int, context_tokens: torch.Tensor) -> torch.Tensor:
		last_token = context_tokens[:, -1:]
		
		context = None
		if Cfg.reset_context_delta and context_tokens.size(1) > 1:
			context = self.gpt.core(context_tokens[:, :-1])["past_key_values"]

		logits_before_shift = self.gpt.core(context_tokens)["logits"][:, -1, :]
		logits_before_shift[:, Cfg.clip_vocab_size:] = -torch.inf
		probs_before_shift = nn.functional.softmax(logits_before_shift, dim=-1)

		if context:
			context = self.shift_context(context, last_token, context_tokens, probs_before_shift)

		lm_output = self.gpt.core(last_token, past_key_values=context)
		logits = lm_output["logits"][:, -1, :]
		logits = self.update_logits(context_tokens, i, logits)
		
		probs = nn.functional.softmax(logits, dim=-1)
		probs = (probs ** Cfg.fusion_factor) * (probs_before_shift ** (1 - Cfg.fusion_factor))
		probs = probs / probs.sum()
		return probs

	def generate_text(self, image_features: torch.Tensor, cond_text: str) -> list[str]:
		self.image_features = image_features
		beam_size = Cfg.beam_size
		
		prompt_tokens = self.tokenizer.encode(self.context_prefix_str + cond_text)
		context_tokens = torch.tensor(prompt_tokens, device=self.device, dtype=torch.long).unsqueeze(0)

		gen_tokens = None
		scores = None
		seq_lengths = torch.ones(beam_size, device=self.device)
		is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)
		
		self.image_features = self.image_features.expand(beam_size, *self.image_features.shape[1:])

		for i in range(Cfg.target_seq_length):
			probs = self.get_next_probs(i, context_tokens)
			logits = probs.log()

			if scores is None:
				scores, next_tokens = logits.topk(beam_size, -1)
				context_tokens = context_tokens.expand(beam_size, *context_tokens.shape[1:])
				next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
				gen_tokens = next_tokens
			else:
				logits[is_stopped] = -torch.inf
				logits[is_stopped, Cfg.eos_token_id] = 0
				scores_sum = scores[:, None] + logits
				seq_lengths[~is_stopped] += 1
				scores_sum_average = scores_sum / seq_lengths[:, None]
				scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
				
				next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='floor')
				seq_lengths = seq_lengths[next_tokens_source]
				next_tokens = next_tokens % scores_sum.shape[1]
				next_tokens = next_tokens.unsqueeze(1)
				gen_tokens = gen_tokens[next_tokens_source]
				gen_tokens = torch.cat((gen_tokens, next_tokens), dim=-1)
				context_tokens = context_tokens[next_tokens_source]
				scores = scores_sum_average * seq_lengths
				is_stopped = is_stopped[next_tokens_source]

			context_tokens = torch.cat((context_tokens, next_tokens), dim=1)
			is_stopped = is_stopped | next_tokens.eq(self.end_token_id).squeeze()

			if is_stopped.all():
				break

		scores = scores / seq_lengths
		output_list = gen_tokens.cpu().numpy()
		output_texts = []
		for output, length in zip(output_list, seq_lengths):
			text = self.tokenizer.decode(output[: int(length)])
			text = text.replace(self.context_prefix_str, '').split(Cfg.end_token)[0]
			output_texts.append(text)

		order = scores.argsort(descending=True)
		output_texts = [output_texts[i] for i in order]
		return output_texts