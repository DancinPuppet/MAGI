from __future__ import annotations
import torch

def standardized(value: torch.Tensor) -> torch.Tensor:
    return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-05)

def q_log_probability(proposal_probability: torch.Tensor, particles: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    probability = proposal_probability.clamp(1e-05, 1.0 - 1e-05)
    values = particles * torch.log(probability[None, :])
    values += (1.0 - particles) * torch.log1p(-probability[None, :])
    return (values * observed[None, :]).sum(dim=1) / observed.sum().clamp_min(1.0)
