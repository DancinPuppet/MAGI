from __future__ import annotations
import argparse
import copy
from typing import Any
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from . import graph_data as base
from .proposal_network import ResidualGraphBlock
from .inference_utils import balanced_log_probability

class ConditionalSourceVAE(nn.Module):

    def __init__(self, graph: base.GraphOperator, hidden: int, layers: int, latent: int, dropout: float) -> None:
        super().__init__()
        self.graph = graph
        self.latent = latent
        self.prior_input = nn.Linear(5, hidden)
        self.posterior_input = nn.Linear(7, hidden)
        self.prior_blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.posterior_blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.prior_head = nn.Linear(3 * hidden, 2 * latent)
        self.posterior_head = nn.Linear(4 * hidden, 2 * latent)
        self.decoder_input = nn.Linear(5 + latent, hidden)
        self.decoder_blocks = nn.ModuleList([ResidualGraphBlock(graph, hidden, dropout) for _ in range(layers)])
        self.latent_query = nn.Linear(latent, hidden)
        self.decoder_output = nn.Linear(hidden, 1)

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (hidden * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _blocks(self, hidden: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            hidden = block(hidden)
        return hidden

    def prior(self, observed: torch.Tensor, proposal: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed_hop = self.graph(observed)
        features = torch.stack([observed, proposal, degree[None, :].expand_as(observed), clustering[None, :].expand_as(observed), observed_hop], dim=-1)
        hidden = self._blocks(nn.functional.gelu(self.prior_input(features)), self.prior_blocks)
        pooled = torch.cat([self._pool(hidden, observed), self._pool(hidden, proposal), hidden.mean(dim=1)], dim=1)
        mean, log_variance = self.prior_head(pooled).chunk(2, dim=1)
        return (mean, log_variance.clamp(-8.0, 5.0))

    def posterior(self, source: torch.Tensor, observed: torch.Tensor, proposal: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observed_hop = self.graph(observed)
        source_hop = self.graph(source)
        features = torch.stack([observed, proposal, source, degree[None, :].expand_as(observed), clustering[None, :].expand_as(observed), observed_hop, source_hop], dim=-1)
        hidden = self._blocks(nn.functional.gelu(self.posterior_input(features)), self.posterior_blocks)
        pooled = torch.cat([self._pool(hidden, observed), self._pool(hidden, proposal), self._pool(hidden, source), hidden.mean(dim=1)], dim=1)
        mean, log_variance = self.posterior_head(pooled).chunk(2, dim=1)
        return (mean, log_variance.clamp(-8.0, 5.0))

    def decode(self, latent: torch.Tensor, observed: torch.Tensor, proposal: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor) -> torch.Tensor:
        observed_hop = self.graph(observed)
        latent_nodes = latent[:, None, :].expand(-1, observed.shape[1], -1)
        features = torch.cat([proposal[..., None], observed[..., None], degree[None, :, None].expand(observed.shape[0], -1, -1), clustering[None, :, None].expand(observed.shape[0], -1, -1), observed_hop[..., None], latent_nodes], dim=-1)
        hidden = self._blocks(nn.functional.gelu(self.decoder_input(features)), self.decoder_blocks)
        latent_query = self.latent_query(latent)[:, None, :]
        interaction = (hidden * latent_query).sum(dim=-1) / hidden.shape[-1] ** 0.5
        residual = self.decoder_output(hidden).squeeze(-1) + interaction
        base_logit = torch.logit(proposal.clamp(1e-05, 1.0 - 1e-05))
        logits = base_logit + residual
        logits = torch.where(observed > 0.5, logits, torch.full_like(logits, -12.0))
        target_mass = proposal.sum(dim=1, keepdim=True)
        offset = torch.zeros_like(target_mass)
        for _ in range(12):
            probability = torch.sigmoid(logits + offset) * observed
            error = probability.sum(dim=1, keepdim=True) - target_mass
            derivative = (probability * (1.0 - probability)).sum(dim=1, keepdim=True).clamp_min(0.0001)
            offset = (offset - error / derivative).clamp(-12.0, 12.0)
        return torch.where(observed > 0.5, logits + offset, torch.full_like(logits, -12.0))

def source_validation(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    threshold = base.choose_threshold(labels, probability)
    prediction = probability >= threshold
    f1 = float(np.mean([f1_score(label, output, zero_division=0) for label, output in zip(labels, prediction)]))
    try:
        auc = float(roc_auc_score(labels.reshape(-1), probability.reshape(-1)))
    except ValueError:
        auc = 0.5
    return (f1, auc, threshold)

def evaluate_scores(graph_nx: Any, labels: np.ndarray, scores: np.ndarray, threshold: float, skip_aed: bool) -> dict[str, float]:
    if not skip_aed:
        return base.evaluate(graph_nx, labels, scores, threshold)
    predictions = scores >= threshold
    f1 = float(np.mean([f1_score(label, output, zero_division=0) for label, output in zip(labels, predictions)]))
    try:
        auc = float(roc_auc_score(labels.reshape(-1), scores.reshape(-1)))
    except ValueError:
        auc = 0.5
    return {'f1': f1, 'auc': auc, 'threshold': float(threshold), 'samples': float(len(labels))}

def train_source_vae(model: ConditionalSourceVAE, train_source: torch.Tensor, train_observed: torch.Tensor, train_q: torch.Tensor, val_source: torch.Tensor, val_observed: torch.Tensor, val_q: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor, args: argparse.Namespace, seed: int) -> tuple[list[dict[str, float]], dict[str, Any]]:
    base.seed_everything(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.source_learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed + 1)
    generator = torch.Generator(device=train_source.device).manual_seed(seed + 2)
    history: list[dict[str, float]] = []
    best: dict[str, Any] | None = None
    stale = 0
    for epoch in range(1, args.source_epochs + 1):
        model.train()
        totals = np.zeros(3)
        batches = 0
        kl_weight = args.source_kl_weight * min(1.0, epoch / max(args.source_kl_warmup, 1))
        for indices in base.batch_indices(len(train_source), args.batch_size, rng):
            index = torch.as_tensor(indices, dtype=torch.long, device=train_source.device)
            source, observed, proposal = (train_source[index], train_observed[index], train_q[index])
            q_mean, q_logvar = model.posterior(source, observed, proposal, degree, clustering)
            p_mean, p_logvar = model.prior(observed, proposal, degree, clustering)
            epsilon = torch.randn(q_mean.shape, generator=generator, device=q_mean.device)
            latent = q_mean + epsilon * torch.exp(0.5 * q_logvar)
            if args.proposal_corruption > 0:
                keep = torch.rand(proposal.shape, generator=generator, device=proposal.device) >= args.proposal_corruption
                fallback = (proposal * observed).sum(dim=1, keepdim=True) / observed.sum(dim=1, keepdim=True).clamp_min(1.0)
                decoder_proposal = torch.where(keep, proposal, fallback * observed)
            else:
                decoder_proposal = proposal
            logits = model.decode(latent, observed, decoder_proposal, degree, clustering)
            reconstruction = -balanced_log_probability(logits, source).mean()
            variance_ratio = torch.exp(q_logvar - p_logvar)
            mean_term = (q_mean - p_mean).square() * torch.exp(-p_logvar)
            kl_elements = 0.5 * (p_logvar - q_logvar + variance_ratio + mean_term - 1.0)
            raw_kl = kl_elements.sum(dim=1).mean()
            free_bits_kl = kl_elements.clamp_min(args.source_free_bits).sum(dim=1).mean()
            loss = reconstruction + kl_weight * free_bits_kl
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [loss.item(), reconstruction.item(), raw_kl.item()]
            batches += 1
        row = {'epoch': epoch, 'loss': float(totals[0] / batches), 'reconstruction': float(totals[1] / batches), 'kl': float(totals[2] / batches)}
        if epoch % args.validation_interval == 0 or epoch == args.source_epochs:
            model.eval()
            with torch.no_grad():
                mean, _ = model.prior(val_observed, val_q, degree, clustering)
                probability = torch.sigmoid(model.decode(mean, val_observed, val_q, degree, clustering)).cpu().numpy()
            f1, auc, threshold = source_validation(val_source.cpu().numpy(), probability)
            row.update({'val_f1': f1, 'val_auc': auc, 'threshold': threshold})
            criterion = (f1, auc)
            if best is None or criterion > best['criterion']:
                best = {'criterion': criterion, 'state': copy.deepcopy(model.state_dict()), 'epoch': epoch, 'val_f1': f1, 'val_auc': auc, 'threshold': threshold}
                stale = 0
            else:
                stale += 1
            if stale >= args.early_stopping_checks:
                history.append(row)
                break
        history.append(row)
    assert best is not None
    model.load_state_dict(best['state'])
    return (history, {k: v for k, v in best.items() if k not in ('state', 'criterion')})
