from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_loss(meg_emb: torch.Tensor, target_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    meg_norm = F.normalize(meg_emb, dim=-1)
    target_norm = F.normalize(target_emb, dim=-1)
    logits = meg_norm @ target_norm.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_mt = F.cross_entropy(logits, labels)
    loss_tm = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_mt + loss_tm)


class SigLIPLoss(nn.Module):
    def __init__(self, temperature_init: float = 10.0, bias_init: float = -10.0) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(temperature_init)).log())
        self.logit_bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, meg_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        meg_norm = F.normalize(meg_emb, dim=-1)
        target_norm = F.normalize(target_emb, dim=-1)
        sim = meg_norm @ target_norm.t()

        scale = self.logit_scale.exp()
        logits = sim * scale + self.logit_bias

        labels = -torch.ones_like(logits)
        labels.fill_(-1.0)
        labels.fill_diagonal_(1.0)
        loss = -F.logsigmoid(labels * logits)
        return loss.mean()


def retrieval_at_k(meg_emb: torch.Tensor, target_emb: torch.Tensor, k: int = 10) -> float:
    meg_norm = F.normalize(meg_emb, dim=-1)
    target_norm = F.normalize(target_emb, dim=-1)
    logits = meg_norm @ target_norm.t()
    topk = logits.topk(k=min(k, logits.shape[1]), dim=1).indices
    labels = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    correct = (topk == labels).any(dim=1).float().mean()
    return float(correct.item())
