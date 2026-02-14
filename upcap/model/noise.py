import math
from os import PathLike

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from upcap.config import Cfg
from utils.dataset import CocoDataset, DType
from utils.logger import logger
from utils.tool import tqdm

BATCH_SIZE = 2048
TRAIN_STEPS = 20000
LR = 1e-4
WEIGHT_DECAY = 1e-4

SINKHORN_EPS = 0.03
SINKHORN_ITERS = 30

SOLVER_STEPS = 20

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.emb_dim // 2
        freq = torch.exp(
            torch.linspace(0, math.log(10000), half, device=t.device, dtype=t.dtype) * (-1)
        )
        phase = t * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            emb = F.pad(emb, (0, self.emb_dim - emb.shape[-1]))
        return emb


class VelocityField(nn.Module):
    def __init__(self, dim: int, t_dim: int = 64, hidden: int = 1024):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + t_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_emb(t)
        x = torch.cat([z_t, t_emb], dim=-1)
        return self.net(x)

    @torch.inference_mode()
    def transport(self, text_tensor: Tensor, steps: int = SOLVER_STEPS) -> Tensor:
        x = F.normalize(text_tensor, dim=-1)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
            v = self(x, t)
            x = x + dt * v
        return F.normalize(x, dim=-1)

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint_path: str | PathLike[str],
        map_location: torch.device | str | None = None,
        t_dim: int = 64,
        hidden: int = 1024,
    ) -> 'VelocityField':
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True
        )
        state_dict = payload['state_dict'] if isinstance(payload, dict) and 'state_dict' in payload else payload
        dim = state_dict['net.4.weight'].shape[0]
        model = cls(dim=dim, t_dim=t_dim, hidden=hidden)
        model.load_state_dict(state_dict)
        if map_location is not None:
            model = model.to(map_location)
        model.eval()
        return model


def sinkhorn_plan(x: Tensor, y: Tensor, eps: float, iters: int) -> Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    cost = 1.0 - x @ y.transpose(0, 1)
    k = torch.exp(-cost / eps).clamp_min(1e-12)

    n, m = x.shape[0], y.shape[0]
    a = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    b = torch.full((m,), 1.0 / m, device=x.device, dtype=x.dtype)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(iters):
        u = a / (k @ v).clamp_min(1e-12)
        v = b / (k.transpose(0, 1) @ u).clamp_min(1e-12)

    plan = (u.unsqueeze(1) * k) * v.unsqueeze(0)
    return plan


@torch.no_grad()
def barycentric_target(plan: Tensor, y: Tensor) -> Tensor:
    row_mass = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
    y_bar = (plan @ y) / row_mass
    return F.normalize(y_bar, dim=-1)


@torch.no_grad()
def load_unpaired_feature_pools() -> tuple[Tensor, Tensor]:
    dataset = CocoDataset(
        annotations=Cfg.coco_train_ann,
        images_path=Cfg.coco_train_image,
        cache_path=Cfg.coco_train_cache,
        dtype=DType.TEXT_FEAT | DType.IMAGE_FEAT,
    )

    text_pool = dataset.texts_feats[::2].float().contiguous()
    image_pool = dataset.images_feats[1::2].float().contiguous()

    text_pool = F.normalize(text_pool, dim=-1)
    image_pool = F.normalize(image_pool, dim=-1)
    return text_pool, image_pool


def sample_unpaired_batch(text_pool: Tensor, image_pool: Tensor, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    n_text = text_pool.shape[0]
    n_img = image_pool.shape[0]

    text_idx = torch.randint(0, n_text, (batch_size,), device=device)
    image_idx = torch.randint(0, n_img, (batch_size,), device=device)

    x = text_pool[text_idx]
    y = image_pool[image_idx]
    return x, y


def train_rectified_flow() -> dict[str, Tensor]:
    device = Cfg.device

    with logger('noise', 'loading feature pools', Cfg.is_master):
        text_pool, image_pool = load_unpaired_feature_pools()
        text_pool = text_pool.to(device, non_blocking=True)
        image_pool = image_pool.to(device, non_blocking=True)

    dim = text_pool.shape[-1]
    model = VelocityField(dim=dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if Cfg.is_master:
        iterator = tqdm(range(TRAIN_STEPS))
    else:
        iterator = range(TRAIN_STEPS)

    for step in iterator:
        x0, y = sample_unpaired_batch(text_pool, image_pool, BATCH_SIZE, device)

        with torch.no_grad():
            plan = sinkhorn_plan(x0, y, eps=SINKHORN_EPS, iters=SINKHORN_ITERS)
            x1 = barycentric_target(plan, y)

        t = torch.rand((BATCH_SIZE, 1), device=device, dtype=x0.dtype)
        z_t = (1.0 - t) * x0 + t * x1
        target_v = x1 - x0

        pred_v = model(z_t, t)
        loss_fm = F.mse_loss(pred_v, target_v)

        with torch.no_grad():
            ot_cost = (plan * (1.0 - x0 @ y.transpose(0, 1))).sum()

        loss = loss_fm

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if Cfg.is_master and (step % 200 == 0 or step == TRAIN_STEPS - 1):
            iterator.set_postfix({
                'loss_fm': f'{(loss_fm.item() * 1e6):.3f}E-6',
                'ot_cost': f'{ot_cost.item():.3f}',
            })

    return model.state_dict()


def main():
    with logger('noise', 'training rectified flow', Cfg.is_master):
        state_dict = train_rectified_flow()

    Cfg.noise_rf_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, Cfg.noise_rf_path)
    if Cfg.is_master:
        print(f'Saved rectified-flow noise model to {Cfg.noise_rf_path}')


if __name__ == '__main__':
    main()
