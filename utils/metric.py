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
        tokenizer: SimpleTokenizer | None = None,
        enable_bleu: bool = True,
        enable_meteor: bool = True,
        enable_cider: bool = True,
        enable_spice: bool = True,
        enable_clip: bool = True,
    ):
        if clip_model is None or preprocess is None:
            clip_model, preprocess = clip.load(Config.clip_pretrained_path, device=Config.device, jit=False)
        if tokenizer is None:
            tokenizer = SimpleTokenizer()
        
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.enable_clip = enable_clip
    
        self.scorers = []
        if enable_bleu:
            self.scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if enable_meteor:
            self.scorers.append((Meteor(), "METEOR"))
        if enable_cider:
            self.scorers.append((Cider(), "CIDEr"))
        if enable_spice:
            self.scorers.append((Spice(), "SPICE"))

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
            
            if method_name == "METEOR":
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

    def _resolve_feat_maps(self, ids, precomputed_tensors: dict | None):
        image_feat_map = None
        ref_feat_map = None

        if precomputed_tensors is None:
            return image_feat_map, ref_feat_map

        image_feat_map = precomputed_tensors.get('image_feats_by_id')
        if image_feat_map is None and 'image_ids' in precomputed_tensors and 'image_feats' in precomputed_tensors:
            image_ids = precomputed_tensors['image_ids']
            image_feats = precomputed_tensors['image_feats']
            image_feat_map = {
                image_id: image_feats[i]
                for i, image_id in enumerate(image_ids)
            }

        ref_feat_map = precomputed_tensors.get('ref_text_feats_by_id')
        if ref_feat_map is None and 'image_ids' in precomputed_tensors and 'ref_text_feats' in precomputed_tensors:
            image_ids = precomputed_tensors['image_ids']
            ref_text_feats = precomputed_tensors['ref_text_feats']
            ref_feat_map = {
                image_id: ref_text_feats[i]
                for i, image_id in enumerate(image_ids)
            }

        if image_feat_map is not None:
            missing = [image_id for image_id in ids if image_id not in image_feat_map]
            if missing:
                raise KeyError(f"Missing image feats for {len(missing)} ids, example: {missing[0]}")

        if ref_feat_map is not None:
            missing = [image_id for image_id in ids if image_id not in ref_feat_map]
            if missing:
                raise KeyError(f"Missing ref text feats for {len(missing)} ids, example: {missing[0]}")

        return image_feat_map, ref_feat_map

    def _compute_clip_metrics(self, gts, preds, tqdm_stream, precomputed_tensors: dict | None = None):
        metric_scores = {}
        clip_s_scores = []
        clip_s_ref_scores = []
        ids = list(gts.keys())
        device = Config.device
        batch_size = 256

        image_feat_map, ref_feat_map = self._resolve_feat_maps(ids, precomputed_tensors)

        if image_feat_map is not None:
            batch_size = 1024
            pbar = tqdm(total=len(ids), desc="CLIP-S/Ref", file=tqdm_stream)

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_preds = [preds[image_id][0] for image_id in batch_ids]
                cand_tokens = clip.tokenize(batch_preds, truncate=True).to(device)

                with torch.no_grad():
                    cand_feats = self.clip_model.encode_text(cand_tokens)
                    cand_feats = cand_feats / cand_feats.norm(dim=-1, keepdim=True)
                    cand_feats = cand_feats.float()

                    image_feats = torch.stack([image_feat_map[image_id] for image_id in batch_ids], dim=0)
                    image_feats = image_feats.to(device).float()
                    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)

                    per_sample_clip_s = 2.5 * torch.clamp(torch.sum(cand_feats * image_feats, dim=1), min=0).cpu().numpy()
                    clip_s_scores.extend(per_sample_clip_s)

                    if ref_feat_map is None:
                        batch_refs = [gts[image_id] for image_id in batch_ids]
                        flat_refs = [r for refs in batch_refs for r in refs]
                        ref_lens = [len(refs) for refs in batch_refs]
                        ref_tokens = clip.tokenize(flat_refs, truncate=True).to(device)
                        all_ref_feats = self.clip_model.encode_text(ref_tokens)
                        all_ref_feats = all_ref_feats / all_ref_feats.norm(dim=-1, keepdim=True)
                        all_ref_feats = all_ref_feats.float()

                        curr_idx = 0
                        for j, length in enumerate(ref_lens):
                            ref_feats = all_ref_feats[curr_idx : curr_idx + length]
                            curr_idx += length
                            sims = torch.matmul(ref_feats, cand_feats[j])
                            ref_score = torch.max(torch.clamp(sims, min=0)).item()
                            clip_s_ref_scores.append(ref_score)
                    else:
                        batch_ref_feats = [ref_feat_map[image_id] for image_id in batch_ids]
                        ref_lens = [rf.shape[0] for rf in batch_ref_feats]
                        total_refs = sum(ref_lens)

                        if total_refs == 0:
                            clip_s_ref_scores.extend([0.0] * len(batch_ids))
                        else:
                            flat_ref_feats = torch.cat(batch_ref_feats, dim=0).to(device).float()
                            flat_ref_feats = flat_ref_feats / flat_ref_feats.norm(dim=-1, keepdim=True)
                            sims_all = torch.matmul(flat_ref_feats, cand_feats.transpose(0, 1))

                            owner = torch.repeat_interleave(
                                torch.arange(len(ref_lens), device=device, dtype=torch.long),
                                torch.tensor(ref_lens, device=device, dtype=torch.long)
                            )
                            row_idx = torch.arange(total_refs, device=device)
                            selected = torch.clamp(sims_all[row_idx, owner], min=0)

                            scores = torch.zeros(len(ref_lens), device=device, dtype=selected.dtype)
                            scores.scatter_reduce_(0, owner, selected, reduce='amax', include_self=True)
                            clip_s_ref_scores.extend(scores.cpu().tolist())

                pbar.update(len(batch_ids))

            pbar.close()
            metric_scores["CLIP-S"] = np.mean(clip_s_scores)
            metric_scores["CLIP-S-Ref"] = np.mean(clip_s_ref_scores)
            return metric_scores

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
                cand_feats = cand_feats.float()
                
                image_feats = self.clip_model.encode_image(images)
                image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
                image_feats = image_feats.float()
                
                per_sample_clip_s = 2.5 * torch.clamp(torch.sum(cand_feats * image_feats, dim=1), min=0).cpu().numpy()
                clip_s_scores.extend(per_sample_clip_s)

                flat_refs = [r for refs in batch_refs for r in refs]
                ref_lens = [len(refs) for refs in batch_refs]
                ref_tokens = clip.tokenize(flat_refs, truncate=True).to(device)
                
                all_ref_feats = self.clip_model.encode_text(ref_tokens)
                all_ref_feats = all_ref_feats / all_ref_feats.norm(dim=-1, keepdim=True)
                all_ref_feats = all_ref_feats.float()
                
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

    def compute(self, ground_truths: dict, predictions: dict, precomputed_tensors: dict | None = None) -> dict:
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
            if self.enable_clip:
                clip_scores = self._compute_clip_metrics(clean_gts, clean_preds, tqdm_stream, precomputed_tensors)
            else:
                clip_scores = {}
            
        return {**trad_scores, **clip_scores}