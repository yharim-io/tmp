import torch
import nltk
from torch import Tensor
from clip.simple_tokenizer import SimpleTokenizer

from yottacap.config import Cfg

class NLTKChunker:
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        # 修正：直接访问 decoder 字典以检测 </w> 后缀
        self.decoder = self.tokenizer.decoder 
        
        grammar = r"""
            NP: {<DT|PP\$>?<JJ>*<NN.*>+}
        """
        self.parser = nltk.RegexpParser(grammar)
        
        # 确保数据存在
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        
    def get_chunk_ids(self, token_ids: Tensor) -> Tensor:
        
        L = token_ids.shape[0]
        # 初始化：默认每个 token 是独立的，end 指向自己
        chunk_ends = torch.arange(L, device=Cfg.device)
        
        # 1. 重建单词并记录 Token 映射关系
        words = []
        word_to_token_indices = [] # 记录每个单词包含的 token 索引列表
        
        current_word_parts = []
        current_token_indices = []
        
        for i, tid in enumerate(token_ids):
            t_item = tid.item()
            
            # 跳过特殊 Token
            if t_item in [0, Cfg.eos_token_id, Cfg.sot_token_id]:
                continue
                
            # 获取原始 BPE token 字符串 (包含 </w>)
            # 注意：需处理潜在的 KeyError，虽然在标准 CLIP 流程中不应发生
            token_str: str = self.decoder.get(t_item, "")
            
            current_word_parts.append(token_str.replace('</w>', ''))
            current_token_indices.append(i)
            
            # 如果以 </w> 结尾，或者是最后一个 token，视为单词结束
            if token_str.endswith('</w>'):
                full_word = "".join(current_word_parts)
                words.append(full_word)
                word_to_token_indices.append(current_token_indices)
                
                # 重置缓冲区
                current_word_parts = []
                current_token_indices = []

        if not words:
            return chunk_ends
        
        # 2. 对完整单词进行 POS Tagging 和 Chunking
        tagged = nltk.pos_tag(words)
        tree = self.parser.parse(tagged)
        
        # 3. 将 Chunk 结果映射回 Token ID
        word_idx = 0
        for subtree in tree:
            if isinstance(subtree, nltk.Tree) and subtree.label() == 'NP':
                num_words_in_phrase = len(subtree.leaves())
                
                # 获取该短语包含的所有 Token 的范围
                # 范围起始：短语第一个单词的第一个 Token
                start_token_idx = word_to_token_indices[word_idx][0]
                # 范围结束：短语最后一个单词的最后一个 Token
                end_token_idx = word_to_token_indices[word_idx + num_words_in_phrase - 1][-1]
                
                # 更新 chunk_ends: 范围内所有 token 的 end 都指向 end_token_idx
                chunk_ends[start_token_idx : end_token_idx + 1] = end_token_idx
                
                word_idx += num_words_in_phrase
            else:
                # 跳过非 NP 的单个单词
                word_idx += 1
                
        return chunk_ends