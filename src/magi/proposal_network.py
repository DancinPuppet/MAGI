from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from . import graph_data as base

@dataclass(frozen=True)
class Profile:
    name: str
    hidden: int
    layers: int
    multiscale: bool
PROFILES = {profile.name: profile for profile in (Profile('res64_l4', 64, 4, False), Profile('multiscale64_l4', 64, 4, True), Profile('multiscale128_l6', 128, 6, True))}

class ResidualGraphBlock(nn.Module):

    def __init__(self, graph: base.GraphOperator, hidden: int, dropout: float) -> None:
        super().__init__()
        self.graph = graph
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        message = self.graph(hidden)
        update = torch.nn.functional.gelu(self.update(torch.cat([hidden, message], dim=-1)))
        return self.norm(hidden + self.dropout(update))

class StrongProposal(nn.Module):

    def __init__(self, graph: base.GraphOperator, hidden: int, layers: int, multiscale: bool, dropout: float) -> None:
        super().__init__()
        self.graph = graph
        self.multiscale = multiscale
        input_dim = 9 if multiscale else 4
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.global_projection = nn.Linear(2 * hidden, hidden)
        self.global_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 1)

    def features(self, observed: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        values = [observed, degree[None, :].expand_as(observed), clustering[None, :].expand_as(observed), memory]
        if self.multiscale:
            hop1 = self.graph(observed)
            hop2 = self.graph(hop1)
            hop4 = self.graph(self.graph(hop2))
            boundary = torch.abs(observed - hop1)
            ratio = observed.mean(dim=1, keepdim=True).expand_as(observed)
            values.extend([hop1, hop2, hop4, boundary, ratio])
        return torch.stack(values, dim=-1)

    def encode(self, observed: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.input(self.features(observed, degree, clustering, memory)))
        for block in self.blocks:
            hidden = block(hidden)
        infected_count = observed.sum(dim=1, keepdim=True).clamp_min(1.0)
        infected_pool = (hidden * observed[..., None]).sum(dim=1) / infected_count
        global_pool = hidden.mean(dim=1)
        context = torch.nn.functional.gelu(self.global_projection(torch.cat([infected_pool, global_pool], dim=-1)))
        return self.global_norm(hidden + context[:, None, :])

    def forward(self, observed: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(observed, degree, clustering, memory)
        return self.output(hidden).squeeze(-1)

def feasible_scores(logits: torch.Tensor, observations: torch.Tensor) -> np.ndarray:
    probability = torch.sigmoid(logits)
    probability = torch.where(observations > 0.5, probability, torch.full_like(probability, 1e-05))
    return probability.detach().cpu().numpy()
