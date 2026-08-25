from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from . import graph_data as base
from .proposal_network import StrongProposal, feasible_scores

def structural_features(observations: np.ndarray, adjacency: np.ndarray, degree: np.ndarray, clustering: np.ndarray) -> np.ndarray:
    denominator = np.maximum(base.adjacency_row_sum(adjacency), 1.0)
    hop1 = base.propagate_features(observations, adjacency) / denominator[None, :]
    hop2 = base.propagate_features(hop1, adjacency) / denominator[None, :]
    hop3 = base.propagate_features(hop2, adjacency) / denominator[None, :]
    boundary = np.abs(observations - hop1)
    ratio = np.broadcast_to(observations.mean(axis=1, keepdims=True), observations.shape)
    degree_values = np.broadcast_to(degree, observations.shape)
    clustering_values = np.broadcast_to(clustering, observations.shape)
    slope = hop1 - hop2
    curvature = hop1 - 2.0 * hop2 + hop3
    redundancy = base.propagate_features(observations * hop1, adjacency) / denominator[None, :]
    return np.stack([observations, hop1, hop2, hop3, boundary, degree_values, clustering_values, ratio, slope, curvature, redundancy], axis=-1).astype(np.float32)

def fit_standardization(train_features: np.ndarray, train_observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = train_features[train_observed > 0.5]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale[scale < 1e-05] = 1.0
    return (mean.astype(np.float32), scale.astype(np.float32))

def patch_entries(features: np.ndarray, observed: np.ndarray, sources: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode, node = np.where(observed > 0.5)
    values = (features[episode, node] - mean) / scale
    labels = sources[episode, node].astype(np.float32)
    context = values[:, [1, 5, 6, 7]].astype(np.float32)
    return (values.astype(np.float32), labels, context, np.stack([episode, node], axis=1))

class StructuralMemoryEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embedding_dim))
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(features), dim=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(features)
        return (embedding, self.classifier(embedding).squeeze(1))

class ConditionalVAE(nn.Module):

    def __init__(self, embedding_dim: int, condition_dim: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(embedding_dim + condition_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(nn.Linear(latent + condition_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, embedding_dim))

    def posterior(self, embedding: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encoder(torch.cat([embedding, condition], dim=1)).chunk(2, dim=1)
        return (mean, log_variance.clamp(-8.0, 4.0))

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.decoder(torch.cat([latent, condition], dim=1)), dim=1)

def condition_tensor(labels: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    return torch.cat([labels[:, None], context], dim=1)

def train_vae(model: ConditionalVAE, embedding: torch.Tensor, labels: torch.Tensor, context: torch.Tensor, args: argparse.Namespace, seed: int) -> list[dict[str, float]]:
    base.seed_everything(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.generator_learning_rate)
    rng = np.random.default_rng(seed + 1)
    generator = torch.Generator(device=embedding.device).manual_seed(seed + 2)
    history = []
    condition = condition_tensor(labels, context)
    for epoch in range(1, args.generator_epochs + 1):
        model.train()
        total = np.zeros(3, dtype=float)
        batches = 0
        for indices in base.batch_indices(len(embedding), args.patch_batch_size, rng):
            index = torch.as_tensor(indices, dtype=torch.long, device=embedding.device)
            mean, log_variance = model.posterior(embedding[index], condition[index])
            epsilon = torch.randn(mean.shape, generator=generator, device=mean.device)
            latent = mean + epsilon * torch.exp(0.5 * log_variance)
            reconstructed = model.decode(latent, condition[index])
            reconstruction = (1.0 - (reconstructed * embedding[index]).sum(dim=1)).mean()
            kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(dim=1).mean()
            weight = args.vae_kl_weight * min(1.0, epoch / max(args.vae_kl_warmup, 1))
            loss = reconstruction + weight * kl
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += [loss.item(), reconstruction.item(), kl.item()]
            batches += 1
        history.append({'epoch': epoch, 'loss': float(total[0] / max(batches, 1)), 'reconstruction': float(total[1] / max(batches, 1)), 'kl': float(total[2] / max(batches, 1))})
    return history

@torch.no_grad()
def generate_vae(model: ConditionalVAE, embedding: torch.Tensor, labels: torch.Tensor, context: torch.Tensor, seed: int) -> torch.Tensor:
    condition = condition_tensor(labels, context)
    mean, log_variance = model.posterior(embedding, condition)
    generator = torch.Generator(device=embedding.device).manual_seed(seed)
    epsilon = torch.randn(mean.shape, generator=generator, device=mean.device)
    return model.decode(mean + epsilon * torch.exp(0.5 * log_variance), condition)

@torch.no_grad()
def density_ratio_memory(query: torch.Tensor, keys: torch.Tensor, labels: torch.Tensor, topk: int, kernel_temperature: float, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive = keys[labels >= 0.5]
    negative = keys[labels < 0.5]
    selected_positive = min(topk, len(positive))
    selected_negative = min(topk, len(negative))
    positive_similarity = topk_inner_product(query, positive, selected_positive, chunk_size)
    negative_similarity = topk_inner_product(query, negative, selected_negative, chunk_size)
    evidence_rows = []
    confidence_rows = []
    for start in range(0, len(query), chunk_size):
        positive_values = positive_similarity[start:start + chunk_size]
        negative_values = negative_similarity[start:start + chunk_size]
        positive_density = torch.logsumexp(positive_values / kernel_temperature, dim=1) - math.log(selected_positive)
        negative_density = torch.logsumexp(negative_values / kernel_temperature, dim=1) - math.log(selected_negative)
        evidence = positive_density - negative_density
        nearest = torch.maximum(positive_values[:, 0], negative_values[:, 0]).clamp(-1.0, 1.0)
        posterior = torch.sigmoid(evidence)
        agreement = torch.abs(2.0 * posterior - 1.0)
        confidence = ((nearest + 1.0) / 2.0).square() * agreement
        evidence_rows.append(evidence)
        confidence_rows.append(confidence)
    return (torch.cat(evidence_rows), torch.cat(confidence_rows))

@torch.no_grad()
def topk_inner_product(query: torch.Tensor, keys: torch.Tensor, topk: int, chunk_size: int, query_episode: torch.Tensor | None=None, key_episode: torch.Tensor | None=None) -> torch.Tensor:
    if topk <= 0 or len(keys) == 0:
        raise ValueError('topk retrieval requires a non-empty key bank')
    topk = min(int(topk), len(keys))
    exact_limit = int(os.environ.get('MAGI_EXACT_RETRIEVAL_LIMIT', '100000'))
    force_exact = os.environ.get('MAGI_FORCE_EXACT_RETRIEVAL', '0') == '1'
    if force_exact or len(keys) <= exact_limit:
        rows = []
        for start in range(0, len(query), chunk_size):
            selected = query[start:start + chunk_size]
            similarity = selected @ keys.T
            if query_episode is not None:
                if key_episode is None:
                    raise ValueError('key_episode is required for episode exclusion')
                similarity.masked_fill_(query_episode[start:start + len(selected), None] == key_episode[None, :], -10000.0)
            rows.append(torch.topk(similarity, topk, dim=1).values)
        return torch.cat(rows)
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('Large MAGI retrieval requires faiss-cpu; install the pinned dependency before running Facebook Company.') from exc
    faiss.omp_set_num_threads(int(os.environ.get('MAGI_FAISS_THREADS', '10')))
    key_values = np.ascontiguousarray(keys.detach().to(device='cpu', dtype=torch.float32).numpy())
    query_values = np.ascontiguousarray(query.detach().to(device='cpu', dtype=torch.float32).numpy())
    index = faiss.IndexHNSWFlat(key_values.shape[1], int(os.environ.get('MAGI_HNSW_M', '32')), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(os.environ.get('MAGI_HNSW_EF_CONSTRUCTION', '80'))
    index.hnsw.efSearch = int(os.environ.get('MAGI_HNSW_EF_SEARCH', '128'))
    index.add(key_values)
    if query_episode is None:
        distances, _ = index.search(query_values, topk)
        return torch.as_tensor(distances, dtype=query.dtype, device=query.device)
    if key_episode is None:
        raise ValueError('key_episode is required for episode exclusion')
    query_episode_np = query_episode.detach().cpu().numpy()
    key_episode_np = key_episode.detach().cpu().numpy()
    candidate_count = min(len(keys), max(topk * 16, int(os.environ.get('MAGI_HNSW_LOO_CANDIDATES', '512'))))
    distances, indices = index.search(query_values, candidate_count)
    output = np.empty((len(query_values), topk), dtype=np.float32)
    missing = []
    for row, episode in enumerate(query_episode_np):
        valid = key_episode_np[indices[row]] != episode
        values = distances[row][valid]
        if len(values) < topk:
            missing.append(row)
        else:
            output[row] = values[:topk]
    if missing:
        missing_tensor = torch.as_tensor(missing, dtype=torch.long, device=query.device)
        exact = topk_inner_product(query[missing_tensor], keys, topk, chunk_size, query_episode=query_episode[missing_tensor], key_episode=key_episode) if len(keys) <= exact_limit else _exact_episode_topk(query[missing_tensor], keys, topk, chunk_size, query_episode[missing_tensor], key_episode)
        output[np.asarray(missing)] = exact.detach().cpu().numpy()
    return torch.as_tensor(output, dtype=query.dtype, device=query.device)

@torch.no_grad()
def _exact_episode_topk(query: torch.Tensor, keys: torch.Tensor, topk: int, chunk_size: int, query_episode: torch.Tensor, key_episode: torch.Tensor) -> torch.Tensor:
    rows = []
    for start in range(0, len(query), chunk_size):
        selected = query[start:start + chunk_size]
        similarity = selected @ keys.T
        similarity.masked_fill_(query_episode[start:start + len(selected), None] == key_episode[None, :], -10000.0)
        rows.append(torch.topk(similarity, topk, dim=1).values)
    return torch.cat(rows)

def scatter_patch_values(metadata: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=np.float32)
    output[metadata[:, 0], metadata[:, 1]] = values
    return output

def evidence_statistics(evidence: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    selected = evidence[observed > 0.5]
    return (float(selected.mean()), float(max(selected.std(), 1e-05)))

def fast_threshold_and_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 37, dtype=np.float32)
    target = labels.astype(bool)[None, :, :]
    prediction = scores[None, :, :] >= thresholds[:, None, None]
    true_positive = np.logical_and(prediction, target).sum(axis=2)
    false_positive = np.logical_and(prediction, np.logical_not(target)).sum(axis=2)
    false_negative = np.logical_and(np.logical_not(prediction), target).sum(axis=2)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0).mean(axis=1)
    selected = int(np.argmax(f1))
    return (float(thresholds[selected]), float(f1[selected]))

def fused_probability(base_probability: np.ndarray, real_evidence: np.ndarray, real_confidence: np.ndarray, generated_evidence: np.ndarray, generated_confidence: np.ndarray, observed: np.ndarray, real_mean: float, real_scale: float, generated_mean: float, generated_scale: float, generation_mix: float, weight: float) -> np.ndarray:
    base_values = np.clip(base_probability, 1e-05, 1.0 - 1e-05)
    logits = np.log(base_values) - np.log1p(-base_values)
    real_normalized = (real_evidence - real_mean) / max(real_scale, 1e-05)
    generated_normalized = (generated_evidence - generated_mean) / max(generated_scale, 1e-05)
    evidence = (1.0 - generation_mix) * real_normalized + generation_mix * generated_normalized
    confidence = (1.0 - generation_mix) * real_confidence + generation_mix * generated_confidence
    logits += weight * confidence * evidence
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    return np.where(observed > 0.5, probability, 1e-05)

def select_fused_weight(labels: np.ndarray, observed: np.ndarray, base_probability: np.ndarray, real_evidence: np.ndarray, real_confidence: np.ndarray, generated_evidence: np.ndarray, generated_confidence: np.ndarray, weight_grid: list[float], generation_mix_grid: list[float]) -> dict[str, float]:
    real_mean, real_scale = evidence_statistics(real_evidence, observed)
    generated_mean, generated_scale = evidence_statistics(generated_evidence, observed)
    options = []
    for generation_mix in generation_mix_grid:
        for weight in weight_grid:
            probability = fused_probability(base_probability, real_evidence, real_confidence, generated_evidence, generated_confidence, observed, real_mean, real_scale, generated_mean, generated_scale, generation_mix, weight)
            threshold, f1 = fast_threshold_and_f1(labels, probability)
            try:
                auc = float(roc_auc_score(labels.reshape(-1), probability.reshape(-1)))
            except ValueError:
                auc = 0.5
            options.append({'weight': float(weight), 'generation_mix': float(generation_mix), 'threshold': float(threshold), 'f1': f1, 'auc': auc, 'real_mean': real_mean, 'real_scale': real_scale, 'generated_mean': generated_mean, 'generated_scale': generated_scale})
    return max(options, key=lambda row: (row['f1'], row['auc'], -row['weight'], -row['generation_mix']))

def find_proposal(root_values: list[Path], scenario: str, seed: int) -> Path:
    for root in root_values:
        path = root / scenario / f'seed{seed}' / 'model.pt'
        if path.exists():
            return path
    raise FileNotFoundError(f'No proposal checkpoint for {scenario} seed {seed}')

def proposal_probability(model: StrongProposal, observed: torch.Tensor, degree: torch.Tensor, clustering: torch.Tensor, memory: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return feasible_scores(model(observed, degree, clustering, memory), observed)
