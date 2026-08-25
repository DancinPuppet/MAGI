from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from . import forward_consistency as forward_lib
from . import sampling_utils as particle_lib

def find_seed_root(roots: list[Path], scenario: str, seed: int, required: str) -> Path:
    for root in roots:
        candidate = root / scenario / f'seed{seed}'
        if (candidate / required).exists():
            return candidate
    raise FileNotFoundError(f'No {required} for {scenario} seed {seed} in {roots}')

@torch.no_grad()
def particle_energies(particles: list[torch.Tensor], q_values: np.ndarray, observations: np.ndarray, forward_model: torch.nn.Module, degree: torch.Tensor, clustering: torch.Tensor, replay_draws: int, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    q_energies: list[torch.Tensor] = []
    forward_energies: list[torch.Tensor] = []
    forward_variances = []
    forward_spreads = []
    for index, (candidate, q_row, observed_row) in enumerate(zip(particles, q_values, observations)):
        q = torch.as_tensor(q_row, dtype=torch.float32, device=candidate.device)
        observed = torch.as_tensor(observed_row, dtype=torch.float32, device=candidate.device)
        q_score = particle_lib.q_log_probability(q, candidate, observed)
        forward_score, variance = forward_lib.predictive_log_score(forward_model, candidate, observed[None, :].expand(candidate.shape[0], -1), degree, clustering, replay_draws, seed + index * 1009)
        q_energies.append(particle_lib.standardized(q_score))
        forward_energies.append(particle_lib.standardized(forward_score))
        forward_variances.append(float(variance.mean().item()))
        forward_spreads.append(float(forward_score.std(unbiased=False).item()))
    return (q_energies, forward_energies, {'forward_replay_variance': float(np.mean(forward_variances)), 'candidate_forward_score_std': float(np.mean(forward_spreads))})

@torch.no_grad()
def aggregate_particles(particles: list[torch.Tensor], q_energies: list[torch.Tensor], forward_energies: list[torch.Tensor] | None, forward_weight: float, temperature: float) -> tuple[np.ndarray, float]:
    outputs = []
    ess = []
    for index, candidate in enumerate(particles):
        energy = q_energies[index]
        if forward_energies is not None and forward_weight > 0:
            energy = energy + forward_weight * forward_energies[index]
        weights = torch.softmax(energy / temperature, dim=0)
        outputs.append((weights[:, None] * candidate).sum(dim=0).cpu().numpy())
        ess.append(float(1.0 / weights.square().sum().item()))
    return (np.asarray(outputs), float(np.mean(ess)))
