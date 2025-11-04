import torch
from torch import nn, Tensor
from torch.nn import functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# 导入 clipcap 模块
from clipcap.config import Config
from clipcap.layer import ClipCaptionModel # 用于类型提示

# --- 关键修复：移除了全局的 CLIP 加载 ---
# 全局的 device, clip_model, preprocess, Tokenizer 已被移除
# 以防止在 'train.py' 导入时导致重复加载和 OOM

# 我们仍然需要一个全局的 Tokenizer (因为它无状态且轻量)
Tokenizer = SimpleTokenizer()
# --- 修复结束 ---


@torch.no_grad
def decode(model: ClipCaptionModel, prefix_embedding: Tensor, device: torch.device) -> str:
	"""
	Greedy-search 解码器 (适配 ClipCap)
	此版本接受已经 project 过的 prefix_embedding
	命名: 'decode' 遵循 decap 规范
	"""
	
	model.eval()
	emb_cat = prefix_embedding # 形状应为 (1, prefix_length, emb_size)
	entry_length = Config.model.max_seq_length
	temperature = 1.0
	tokens = None
	
	for _ in range(entry_length):
		
		logits = model.gpt.forward_embeds(inputs_embeds=emb_cat)
		logits = logits[:, -1, :] / temperature
		logits = F.softmax(logits, dim=1)
		next_token_id = torch.argmax(logits, -1).unsqueeze(0)
		
		next_token_embed = model.gpt.embed(next_token_id)
		
		if tokens is None:
			tokens = next_token_id
		else:
			tokens = torch.cat((tokens, next_token_id), dim=1)
		
		if next_token_id.item() == Config.model.eos_token_id:
			break
		
		emb_cat = torch.cat((emb_cat, next_token_embed), dim=1)
	
	try:
		output_list = list(tokens.squeeze().cpu().numpy())
		output = Tokenizer.decode(output_list)
		output = output.replace('<|startoftext|>','').replace('<|endoftext|>','')
	except:
		output = 'error'
	
	return output

@torch.no_grad
def generate_beam(*args, **kwargs):
	"""
	占位符。decap 模式不使用 beam search。
	"""
	raise NotImplementedError("Beam search is not implemented in decap-style refactor")


@torch.no_grad
def image_to_text(clipcap_model: ClipCaptionModel, image_path: Path, device: torch.device) -> str:
	"""
	图像到文本的推理接口 (最终版)
	
	关键修复：CLIP 模型现在在此函数内部加载，
	仅在被 'inference.py' 调用时加载，而不是在 'train.py' 导入时。
	"""
	
	# --- 修复：按需加载 CLIP ---
	# 注意: 这假定 clip_model 和 preprocess 尚未加载
	# 在 inference.py 中，我们应该只加载一次
	try:
		# 尝试访问一个已加载的全局模型 (如果 inference.py 提供了它)
		clip_model = globals()["clip_model_global"]
		preprocess = globals()["preprocess_global"]
	except KeyError:
		# 如果不存在，则加载它 (用于独立测试)
		print(f"[decode.py] 正在加载 CLIP (device={device})...")
		clip_model, preprocess = clip.load(Config.model.clip_model_type, device=device, jit=False)
		globals()["clip_model_global"] = clip_model
		globals()["preprocess_global"] = preprocess
	# --- 修复结束 ---

	clipcap_model.eval()
	
	image = Image.open(image_path)
	image_preprocessed = preprocess(image).unsqueeze(0).to(device)
	
	with torch.no_grad():
		image_feature: Tensor = clip_model.encode_image(image_preprocessed).float()
		
		# 使用 clipcap 的 mapper (clip_project) 生成 prefix
		prefix_embedding = clipcap_model.clip_project(image_feature)
		prefix_embedding = prefix_embedding.reshape(1, Config.model.prefix_length, -1) # 形状 (1, 10, 768)
	
	# 调用我们专为 clipcap 修改的 decode 函数
	text = decode(clipcap_model, prefix_embedding, device)
	return text

