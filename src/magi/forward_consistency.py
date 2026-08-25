from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import structural_memory as dual
from . import graph_data as base
from . import retrieval_features as rag
from . import source_posterior as source_vae_lib
from .proposal_network import PROFILES, ResidualGraphBlock, StrongProposal
from .inference_utils import balanced_log_probability, gaussian_kl

def source_preserving_logits(raw: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    probability = source + (1.0 - source) * torch.sigmoid(raw)
    return torch.logit(probability.clamp(1e-05, 1.0 - 1e-05))

class SpectralScalarMLP(nn.Module):

    def __init__(self, hidden: int, contraction: float):
        super().__init__()
        self.first = nn.utils.parametrizations.spectral_norm(nn.Linear(1, hidden))
        self.second = nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, 1))
        self.contraction = contraction

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.contraction * self.second(nn.functional.gelu(self.first(values)))

class InvertibleMeanCore(nn.Module):

    def __init__(self, graph: base.GraphOperator, hidden: int, contraction: float=0.65, propagation_scale: float=0.2) -> None:
        super().__init__()
        self.graph = graph
        self.f = SpectralScalarMLP(hidden, contraction)
        self.g = SpectralScalarMLP(hidden, contraction)
        self.propagation_scale = propagation_scale
        self.calibration = nn.Linear(1, 1)

    def state(self, source: torch.Tensor) -> torch.Tensor:
        values = source[..., None]
        attributed = 0.5 * (values + self.f(values))
        propagated = self.graph(attributed)
        state = 0.5 * (attributed + self.g(attributed) + self.propagation_scale * propagated)
        return state.squeeze(-1)

    def logits(self, source: torch.Tensor) -> torch.Tensor:
        raw = self.calibration(self.state(source)[..., None]).squeeze(-1)
        return source_preserving_logits(raw, source)

class InvertibleResidualCVAE(nn.Module):
    stochastic = True

    def __init__(self, graph: base.GraphOperator, hidden: int, layers: int, latent: int, dropout: float, mean_loss_weight: float, residual_penalty: float) -> None:
        super().__init__()
        self.graph = graph
        self.latent = latent
        self.mean_loss_weight = mean_loss_weight
        self.residual_penalty = residual_penalty
        self.mean_core = InvertibleMeanCore(graph, hidden)
        self.posterior_input = nn.Linear(7, hidden)
        self.prior_input = nn.Linear(4, hidden)
        self.posterior_blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.prior_blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.posterior_head = nn.Linear(3 * hidden, 2 * latent)
        self.prior_head = nn.Linear(2 * hidden, 2 * latent)
        self.decoder_input = nn.Linear(4 + latent, hidden)
        self.decoder_blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.decoder_output = nn.Linear(hidden, 1)

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (hidden * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _blocks(self, hidden: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            hidden = block(hidden)
        return hidden

    def posterior(self, source, observed, degree, clustering):
        source_hop = self.graph(source)
        observed_hop = self.graph(observed)
        features = torch.stack([source, observed, degree[None, :].expand_as(source), clustering[None, :].expand_as(source), source_hop, observed_hop, torch.abs(observed - observed_hop)], dim=-1)
        hidden = self._blocks(nn.functional.gelu(self.posterior_input(features)), self.posterior_blocks)
        pooled = torch.cat([self._pool(hidden, source), self._pool(hidden, observed), hidden.mean(dim=1)], dim=1)
        mean, log_variance = self.posterior_head(pooled).chunk(2, dim=1)
        return (mean, log_variance.clamp(-8.0, 6.0))

    def prior(self, source, degree, clustering):
        exposure = self.graph(source)
        features = torch.stack([source, degree[None, :].expand_as(source), clustering[None, :].expand_as(source), exposure], dim=-1)
        hidden = self._blocks(nn.functional.gelu(self.prior_input(features)), self.prior_blocks)
        pooled = torch.cat([self._pool(hidden, source), hidden.mean(dim=1)], dim=1)
        mean, log_variance = self.prior_head(pooled).chunk(2, dim=1)
        return (mean, log_variance.clamp(-8.0, 6.0))

    def decode(self, source, latent, degree, clustering):
        exposure = self.graph(source)
        latent_nodes = latent[:, None, :].expand(-1, source.shape[1], -1)
        features = torch.cat([source[..., None], degree[None, :, None].expand(source.shape[0], -1, -1), clustering[None, :, None].expand(source.shape[0], -1, -1), exposure[..., None], latent_nodes], dim=-1)
        hidden = self._blocks(nn.functional.gelu(self.decoder_input(features)), self.decoder_blocks)
        residual = self.decoder_output(hidden).squeeze(-1)
        mean_logits = self.mean_core.logits(source)
        return (source_preserving_logits(mean_logits + residual, source), residual)

    def training_loss(self, source, observed, degree, clustering, kl_weight, generator):
        q_mean, q_logvar = self.posterior(source, observed, degree, clustering)
        p_mean, p_logvar = self.prior(source, degree, clustering)
        epsilon = torch.randn(q_mean.shape, generator=generator, device=q_mean.device)
        latent = q_mean + epsilon * torch.exp(0.5 * q_logvar)
        logits, residual = self.decode(source, latent, degree, clustering)
        reconstruction = -balanced_log_probability(logits, observed).mean()
        kl = gaussian_kl(q_mean, q_logvar, p_mean, p_logvar).mean()
        mean_reconstruction = -balanced_log_probability(self.mean_core.logits(source), observed).mean()
        residual_l2 = residual.square().mean()
        loss = reconstruction + kl_weight * kl + self.mean_loss_weight * mean_reconstruction + self.residual_penalty * residual_l2
        return (loss, {'reconstruction': float(reconstruction.item()), 'kl': float(kl.item()), 'mean_reconstruction': float(mean_reconstruction.item()), 'residual_l2': float(residual_l2.item())})

    def sample_logits(self, source, repeats, degree, clustering, generator):
        mean, log_variance = self.prior(source, degree, clustering)
        epsilon = torch.randn(source.shape[0], repeats, self.latent, generator=generator, device=source.device)
        latent = mean[:, None, :] + epsilon * torch.exp(0.5 * log_variance[:, None, :])
        expanded = source[:, None, :].expand(-1, repeats, -1).reshape(-1, source.shape[1])
        logits, _ = self.decode(expanded, latent.reshape(-1, self.latent), degree, clustering)
        return logits.reshape(source.shape[0], repeats, source.shape[1])

def predictive_log_score(model: nn.Module, source: torch.Tensor, observed: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor, repeats: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=source.device).manual_seed(seed)
    logits = model.sample_logits(source, repeats, degree, clustering, generator)
    expanded_observed = observed[:, None, :].expand_as(logits)
    score = balanced_log_probability(logits.reshape(-1, logits.shape[-1]), expanded_observed.reshape(-1, expanded_observed.shape[-1])).reshape(source.shape[0], repeats)
    log_score = torch.logsumexp(score, dim=1) - math.log(repeats)
    probability = torch.sigmoid(logits)
    replay_variance = probability.var(dim=1, unbiased=False).mean(dim=1)
    return (log_score, replay_variance)

def make_rag_proposal_scores(graph: base.GraphOperator, adjacency: np.ndarray, degree_np: np.ndarray, clustering_np: np.ndarray, sources_np: np.ndarray, observations_np: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, checkpoint_path: Path, device: torch.device, dropout: float, topk: int) -> np.ndarray:
    profile = PROFILES['multiscale64_l4']
    proposal = StrongProposal(graph, profile.hidden, profile.layers, profile.multiscale, dropout).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    proposal.load_state_dict(checkpoint.get('model', checkpoint))
    all_scores = np.zeros_like(observations_np, dtype=np.float32)
    train_observations = observations_np[train_idx]
    train_sources = sources_np[train_idx]
    for indices in (train_idx, val_idx, test_idx):
        memory = rag.build_local_memory(train_observations, train_sources, observations_np[indices], adjacency, degree_np, topk)
        all_scores[indices] = dual.proposal_probability(proposal, torch.as_tensor(observations_np[indices], dtype=torch.float32, device=device), torch.as_tensor(degree_np, dtype=torch.float32, device=device), torch.as_tensor(clustering_np, dtype=torch.float32, device=device), torch.as_tensor(memory, dtype=torch.float32, device=device))
    del proposal
    return all_scores

@torch.no_grad()
def source_particles(model: source_vae_lib.ConditionalSourceVAE, proposal_scores: np.ndarray, observations: np.ndarray, degree: torch.Tensor, clustering: torch.Tensor, particles: int, seed: int, device: torch.device) -> list[torch.Tensor]:
    outputs = []
    for index, (q_values, y_values) in enumerate(zip(proposal_scores, observations)):
        q = torch.as_tensor(q_values, dtype=torch.float32, device=device)[None, :]
        observed = torch.as_tensor(y_values, dtype=torch.float32, device=device)[None, :]
        mean, log_variance = model.prior(observed, q, degree, clustering)
        generator = torch.Generator(device=device).manual_seed(seed + index * 1009)
        epsilon = torch.randn(particles, model.latent, generator=generator, device=device)
        latent = mean.expand(particles, -1) + epsilon * torch.exp(0.5 * log_variance.expand(particles, -1))
        logits = model.decode(latent, observed.expand(particles, -1), q.expand(particles, -1), degree, clustering)
        outputs.append(torch.sigmoid(logits) * observed)
    return outputs
