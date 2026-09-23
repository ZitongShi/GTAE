from __future__ import annotations

import torch
from torch import Tensor


def accuracy(logits: Tensor, labels: Tensor, mask: Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    if labels.numel() == 0:
        return 0.0
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def attack_success_rate(clean_logits: Tensor, attacked_logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    clean_correct = clean_logits.argmax(dim=-1).eq(labels)
    attacked_wrong = attacked_logits.argmax(dim=-1).ne(labels)
    eligible = mask & clean_correct
    if not eligible.any():
        return 0.0
    return float(attacked_wrong[eligible].float().mean().item())


def prediction_consistency(clean_logits: Tensor, attacked_logits: Tensor, mask: Tensor) -> Tensor:
    clean = clean_logits[mask].softmax(dim=-1)
    attacked = attacked_logits[mask].softmax(dim=-1)
    if clean.numel() == 0:
        return torch.ones((), device=clean_logits.device)
    return torch.nn.functional.cosine_similarity(clean, attacked, dim=-1).mean()

