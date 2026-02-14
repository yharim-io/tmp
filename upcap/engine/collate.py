import torch
from torch import Tensor
import clip
import random

from upcap.model.parser import TextParser
from upcap.config import Cfg

class CollateFn:
    def __init__(self):
        self.parser = TextParser()
        self._concept_cache: dict[str, list[str]] = {}
        self._expand_cache: dict[str, list[str]] = {}
        self.templates: list[str] = [
            'a photo of {}.',
            'an image of {}.',
            'a picture of {}.',
            'a cutout of {}.',
            'a photo of {} isolated on a black background.',
            'an image of {} isolated on a black background.',
            'a picture of {} isolated on a black background.',
            'a cutout of {} isolated on a black background.',
        ]
        self.num_map: dict[str, int] = {
            # 'a': 1, 'an': 1, 'one': 1, 'single': 1,
            'two': 2, 'couple': 2, 'pair': 2, 'both': 2,
            'three': 3, 'trio': 3, 'triple': 3,
            'four': 4, 'quartet': 4, 'quadruple': 4,
            # 'five': 5, 'quintet': 5, 'quintuple': 5
        }

    def _expand_concept(self, concept: str) -> list[str]:

        cached = self._expand_cache.get(concept)
        if cached is not None:
            return cached
    
        count = 1
        remove_idx = -1

        doc = self.parser.nlp(concept)
        
        for token in doc:
            if token.dep_ == 'nummod' and token.head.tag_ == 'NNS':
                t_lower = token.text.lower()
                if t_lower in self.num_map:
                    count = self.num_map[t_lower]
                    remove_idx = token.i
                elif t_lower.isdigit():
                    count = int(t_lower)
                    remove_idx = token.i
                break
        
        if count > 4 or remove_idx == -1:
            result = [concept]
            self._expand_cache[concept] = result
            return result
        
        tokens = [t.text for t in doc if t.i != remove_idx]
        concept = " ".join(tokens).strip()
        concept = concept.replace("  ", " ")

        result = [concept] * count
        self._expand_cache[concept] = result
        return result

    def __call__(self, batch_data):
        raw_texts: list[str] = [item['text'] for item in batch_data]

        refined_concepts_batch: list[list[str]] = []
        token_counts: list[int] = []

        for text in raw_texts:
            concept_strings = self._concept_cache.get(text)
            if concept_strings is None:
                concept_strings = self.parser(text)
                self._concept_cache[text] = concept_strings

            refined_concepts = [concept_strings[0]] + [
                self.templates[random.randrange(len(self.templates))].format(ec)
                for c in concept_strings[1:]
                for ec in self._expand_concept(c)
            ]
            refined_concepts_batch.append(refined_concepts)
            token_counts.append(len(refined_concepts))

        flat_refined_concepts = [c for concepts in refined_concepts_batch for c in concepts]
        flat_tokens: Tensor = clip.tokenize(flat_refined_concepts, truncate=True)
        token_splits = flat_tokens.split(token_counts, dim=0)

        batch_concepts_list: list[Tensor] = list(token_splits)
        batch_captions_list: list[Tensor] = [tokens[0] for tokens in token_splits]

        batch_size = len(batch_data)
        
        padded_caption_batch: Tensor = torch.zeros(batch_size, Cfg.max_seq_length, dtype=torch.long)

        for i, sample_caption in enumerate(batch_captions_list):
            num_tokens = min(len(sample_caption), Cfg.max_seq_length)
            padded_caption_batch[i, :num_tokens] = sample_caption[:num_tokens]

        return {
            'text_concept_tokens': batch_concepts_list,
            'token_ids': padded_caption_batch
        }

class GlobalCollateFn:
    def __call__(self, batch_data):
        raw_texts = [item['text'] for item in batch_data]
        batch_size = len(raw_texts)
        
        # [batch_size, 77]
        # 直接对整个 batch 进行 tokenize，不需要循环，效率更高
        global_tokens = clip.tokenize(raw_texts, truncate=True)
        
        # 1. Concept: 仅填充第一个位置 (index 0) 为 global tokens
        padded_concept_batch = torch.zeros(batch_size, Cfg.max_concepts, 77, dtype=torch.long)
        padded_concept_batch[:, 0] = global_tokens
        
        # 2. Caption: 截取或填充至 max_seq_length
        padded_caption_batch = torch.zeros(batch_size, Cfg.max_seq_length, dtype=torch.long)
        valid_len = min(77, Cfg.max_seq_length)
        padded_caption_batch[:, :valid_len] = global_tokens[:, :valid_len]

        return {
            'text_concept_tokens': list(padded_concept_batch),
            'token_ids': padded_caption_batch
        }

if __name__ == '__main__':
    collate_fn = CollateFn()
    sample_texts = [
        'there is a choumie and two dabian in the park.',
        'a group of three people are standing together.',
        'an airplane is flying in the sky.',
        'the table has five apples and a banana on it.',
    ]

    for text in sample_texts:
        concepts = collate_fn.parser(text)
        print(concepts)
        for c in concepts[1:]:
            expanded_concepts = collate_fn._expand_concept(c)
            print(expanded_concepts)
        print()