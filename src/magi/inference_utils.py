from __future__ import annotations
import torch
from torch import nn

def balanced_log_probability(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pointwise = target * nn.functional.logsigmoid(logits)
    pointwise += (1.0 - target) * nn.functional.logsigmoid(-logits)
    positive = (pointwise * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
    negative_mask = 1.0 - target
    negative = (pointwise * negative_mask).sum(dim=1) / negative_mask.sum(dim=1).clamp_min(1.0)
    return 0.5 * (positive + negative)

def gaussian_kl(q_mean: torch.Tensor, q_log_variance: torch.Tensor, p_mean: torch.Tensor, p_log_variance: torch.Tensor) -> torch.Tensor:
    variance_ratio = torch.exp(q_log_variance - p_log_variance)
    mean_term = (q_mean - p_mean).square() * torch.exp(-p_log_variance)
    value = p_log_variance - q_log_variance + variance_ratio + mean_term - 1.0
    return 0.5 * value.sum(dim=1)
