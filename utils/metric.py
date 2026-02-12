import os
import torch
import numpy as np
import clip
from PIL import Image
from clip.model import CLIP
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from utils.config import Config
from utils.tool import tqdm

class CLIPDataset(Dataset):
    def __init__(self, ids, gts, preds, image_dir, preprocess):
        self.ids = ids
        self.gts = gts
        self.preds = preds
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.image_dir, img_id)
        image = self.preprocess(Image.open(path))
        return image, self.preds[img_id][0], self.gts[img_id]

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    preds = [item[1] for item in batch]
    refs = [item[2] for item in batch]
    return images, preds, refs

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
    
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

    def _suppress_output(self):
        class Suppress:
            def __enter__(self):
                self.null_fd = os.open(os.devnull, os.O_RDWR)
                self.save_stdout = os.dup(1)
                self.save_stderr = os.dup(2)
                os.dup2(self.null_fd, 1)
                os.dup2(self.null_fd, 2)
            def __exit__(self, *args):
                os.dup2(self.save_stdout, 1)
                os.dup2(self.save_stderr, 2)
                os.close(self.null_fd)
                os.close(self.save_stdout)
                os.close(self.save_stderr)
        return Suppress()

    def _clean_data(self, data: dict) -> dict:
        return {k: [s.replace('\n', ' ').replace('\r', ' ') for s in v] for k, v in data.items()}

    def _wrap_progress(self, data, pbar):
        class ProgressAdapter:
            def __init__(self, data, pbar):
                self.data = data
                self.pbar = pbar
            def __getitem__(self, key):
                self.pbar.update()
                return self.data[key]
            def __len__(self):
                return len(self.data)
            def keys(self):
                return self.data.keys()
        return ProgressAdapter(data, pbar)

    def _compute_batched(self, scorer, gts, preds, pbar):
        scores_list = []
        ids = list(gts.keys())
        batch_size = 2048
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            gts_batch = {k: gts[k] for k in batch_ids}
            res_batch = {k: preds[k] for k in batch_ids}
            
            with self._suppress_output():
                _, scores_each = scorer.compute_score(gts_batch, res_batch)
            
            if isinstance(scorer, Spice):
                scores_list.extend([s['All']['f'] for s in scores_each])
            else:
                scores_list.extend(scores_each)
                
            pbar.update(len(batch_ids))
            
        return np.mean(scores_list)

    def _compute_hooked(self, scorer, gts, preds, pbar):
        gts_wrapped = self._wrap_progress(gts, pbar)
        with self._suppress_output():
            score, _ = scorer.compute_score(gts_wrapped, preds)
        return score

    def _compute_traditional_metrics(self, gts, preds, tqdm_stream):
        metric_scores = {}
        for scorer, method_name in self.scorers:
            display_name = method_name if not isinstance(method_name, list) else "Bleu"
            pbar = tqdm(total=len(gts), desc=display_name, file=tqdm_stream)
            
            if method_name == "SPICE" or method_name == "METEOR":
                score = self._compute_batched(scorer, gts, preds, pbar)
            else:
                score = self._compute_hooked(scorer, gts, preds, pbar)
            
            pbar.close()

            if isinstance(method_name, list):
                for m, s in zip(method_name, score):
                    metric_scores[m] = s
            else:
                metric_scores[method_name] = score
        return metric_scores

    def _compute_clip_metrics(self, gts, preds, tqdm_stream):
        metric_scores = {}
        clip_s_scores = []
        clip_s_ref_scores = []
        ids = list(gts.keys())
        device = Config.device
        batch_size = 256

        dataset = CLIPDataset(ids, gts, preds, Config.coco_val_image, self.preprocess)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True,
            collate_fn=custom_collate
        )

        pbar = tqdm(total=len(ids), desc="CLIP-S/Ref", file=tqdm_stream)
        
        for images, batch_preds, batch_refs in dataloader:
            
            images = images.to(device)
            cand_tokens = clip.tokenize(batch_preds, truncate=True).to(device)
            
            with torch.no_grad():
                cand_feats = self.clip_model.encode_text(cand_tokens)
                cand_feats = cand_feats / cand_feats.norm(dim=-1, keepdim=True)
                
                image_feats = self.clip_model.encode_image(images)
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
                
                per_sample_clip_s = 2.5 * torch.clamp(torch.sum(cand_feats * image_feats, dim=1), min=0).cpu().numpy()
                clip_s_scores.extend(per_sample_clip_s)

                # Batch process references to avoid inner loop overhead
                # Flatten all references
                flat_refs = [r for refs in batch_refs for r in refs]
                ref_lens = [len(refs) for refs in batch_refs]
                ref_tokens = clip.tokenize(flat_refs, truncate=True).to(device)
                
                all_ref_feats = self.clip_model.encode_text(ref_tokens)
                all_ref_feats = all_ref_feats / all_ref_feats.norm(dim=-1, keepdim=True)
                
                curr_idx = 0
                for j, length in enumerate(ref_lens):
                    ref_feats = all_ref_feats[curr_idx : curr_idx + length]
                    curr_idx += length
                    
                    sims = torch.matmul(ref_feats, cand_feats[j])
                    ref_score = torch.max(torch.clamp(sims, min=0)).item()
                    clip_s_ref_scores.append(ref_score)

            pbar.update(len(batch_preds))
        
        pbar.close()
        metric_scores["CLIP-S"] = np.mean(clip_s_scores)
        metric_scores["CLIP-S-Ref"] = np.mean(clip_s_ref_scores)
        
        return metric_scores

    def compute(self, ground_truths: dict, predictions: dict) -> dict:
        clean_gts = self._clean_data(ground_truths)
        clean_preds = self._clean_data(predictions)
        
        gts_for_ptb = {
            k: [{'caption': c} for c in v] 
            for k, v in clean_gts.items()
        }
        preds_for_ptb = {
            k: [{'caption': c} for c in v] 
            for k, v in clean_preds.items()
        }

        tokenizer = PTBTokenizer()
        ptb_clean_gts = tokenizer.tokenize(gts_for_ptb)
        ptb_clean_preds = tokenizer.tokenize(preds_for_ptb)

        tqdm_fd = os.dup(2)
        with os.fdopen(tqdm_fd, 'w') as tqdm_stream:
            trad_scores = self._compute_traditional_metrics(ptb_clean_gts, ptb_clean_preds, tqdm_stream)
            clip_scores = self._compute_clip_metrics(clean_gts, clean_preds, tqdm_stream)
            
        return {**trad_scores, **clip_scores}