from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from . import forward_consistency as forward_lib
from . import graph_data as base
from . import posterior_reweighting as joint_lib
from . import source_posterior as source_lib
from . import structural_memory as memory_lib
from .config import parse_args

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')

def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

@torch.no_grad()
def density_ratio_memory_excluding_episode(query: torch.Tensor, query_episode: torch.Tensor, keys: torch.Tensor, labels: torch.Tensor, key_episode: torch.Tensor, topk: int, kernel_temperature: float, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive_mask = labels >= 0.5
    positive, positive_episode = (keys[positive_mask], key_episode[positive_mask])
    negative, negative_episode = (keys[~positive_mask], key_episode[~positive_mask])
    selected_positive = min(topk, len(positive))
    selected_negative = min(topk, len(negative))
    if selected_positive == 0 or selected_negative == 0:
        raise RuntimeError('Both source and non-source memory keys are required')
    positive_values_all = memory_lib.topk_inner_product(query, positive, selected_positive, chunk_size, query_episode=query_episode, key_episode=positive_episode)
    negative_values_all = memory_lib.topk_inner_product(query, negative, selected_negative, chunk_size, query_episode=query_episode, key_episode=negative_episode)
    evidence_rows, confidence_rows = ([], [])
    for start in range(0, len(query), chunk_size):
        positive_values = positive_values_all[start:start + chunk_size]
        negative_values = negative_values_all[start:start + chunk_size]
        positive_density = torch.logsumexp(positive_values / kernel_temperature, dim=1) - math.log(selected_positive)
        negative_density = torch.logsumexp(negative_values / kernel_temperature, dim=1) - math.log(selected_negative)
        evidence = positive_density - negative_density
        nearest = torch.maximum(positive_values[:, 0], negative_values[:, 0]).clamp(-1.0, 1.0)
        posterior = torch.sigmoid(evidence)
        confidence = ((nearest + 1.0) / 2.0).square() * torch.abs(2.0 * posterior - 1.0)
        evidence_rows.append(evidence)
        confidence_rows.append(confidence)
    return (torch.cat(evidence_rows), torch.cat(confidence_rows))

@torch.no_grad()
def filter_generated_with_indices(encoder: memory_lib.StructuralMemoryEncoder, generated: torch.Tensor, labels: torch.Tensor, minimum_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    logits = encoder.classifier(generated).squeeze(1)
    consistent = (logits >= 0) == (labels >= 0.5)
    if consistent.float().mean().item() < minimum_fraction:
        signed = torch.where(labels >= 0.5, logits, -logits)
        count = max(2, int(math.ceil(minimum_fraction * len(labels))))
        selected = torch.topk(signed, count).indices
    else:
        selected = torch.where(consistent)[0]
    selected_labels = labels[selected]
    if selected_labels.min() == selected_labels.max():
        selected = torch.arange(len(labels), device=labels.device)
        selected_labels = labels
    return (generated[selected], selected_labels, selected, {'raw_count': len(labels), 'kept_count': len(selected), 'classifier_consistency': float(consistent.float().mean().item())})

def scatter(metadata: np.ndarray, values: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    return memory_lib.scatter_patch_values(metadata, values.cpu().numpy(), shape)

def build_memory_conditioned_q(scenario: str, seed: int, memory_root: Path, adjacency: np.ndarray, degree_np: np.ndarray, clustering_np: np.ndarray, sources_np: np.ndarray, observations_np: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, q_direct: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    features = memory_lib.structural_features(observations_np, adjacency, degree_np, clustering_np)
    standard_mean, standard_scale = memory_lib.fit_standardization(features[train_idx], observations_np[train_idx])
    entries = {split: memory_lib.patch_entries(features[indices], observations_np[indices], sources_np[indices], standard_mean, standard_scale) for split, indices in (('train', train_idx), ('val', val_idx), ('test', test_idx))}
    to_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    train_features_np, train_labels_np, train_context_np, train_metadata = entries['train']
    val_features_np, val_labels_np, _, val_metadata = entries['val']
    test_features_np, _, _, test_metadata = entries['test']
    train_features = to_tensor(train_features_np)
    train_labels = to_tensor(train_labels_np)
    train_context = to_tensor(train_context_np)
    val_features = to_tensor(val_features_np)
    test_features = to_tensor(test_features_np)
    encoder = memory_lib.StructuralMemoryEncoder(train_features.shape[1], args.memory_hidden, args.memory_embedding_dim, args.dropout).to(device)
    encoder.load_state_dict(torch.load(memory_root / 'structural_encoder.pt', map_location=device, weights_only=False))
    encoder.eval()
    with torch.no_grad():
        train_embedding = encoder.encode(train_features)
        val_embedding = encoder.encode(val_features)
        test_embedding = encoder.encode(test_features)
    base.seed_everything(seed + 200)
    vae = memory_lib.ConditionalVAE(args.memory_embedding_dim, 1 + train_context.shape[1], args.memory_hidden, args.memory_latent_dim).to(device)
    started = time.perf_counter()
    history = memory_lib.train_vae(vae, train_embedding, train_labels, train_context, args, seed + 201)
    vae.eval()
    generated = memory_lib.generate_vae(vae, train_embedding, train_labels, train_context, seed + 202)
    generated, generated_labels, generated_indices, generation_audit = filter_generated_with_indices(encoder, generated, train_labels, args.minimum_generated_fraction)
    generation_seconds = time.perf_counter() - started
    train_episode = torch.as_tensor(train_metadata[:, 0], dtype=torch.long, device=device)
    generated_episode = train_episode[generated_indices]
    real_train = density_ratio_memory_excluding_episode(train_embedding, train_episode, train_embedding, train_labels, train_episode, args.memory_topk, args.kernel_temperature, args.retrieval_chunk_size)
    generated_train = density_ratio_memory_excluding_episode(train_embedding, train_episode, generated, generated_labels, generated_episode, args.memory_topk, args.kernel_temperature, args.retrieval_chunk_size)

    def ordinary(query: torch.Tensor, metadata: np.ndarray, keys: torch.Tensor, labels: torch.Tensor, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        evidence, confidence = memory_lib.density_ratio_memory(query, keys, labels, args.memory_topk, args.kernel_temperature, args.retrieval_chunk_size)
        return (scatter(metadata, evidence, shape), scatter(metadata, confidence, shape))
    train_real_evidence = scatter(train_metadata, real_train[0], observations_np[train_idx].shape)
    train_real_confidence = scatter(train_metadata, real_train[1], observations_np[train_idx].shape)
    train_generated_evidence = scatter(train_metadata, generated_train[0], observations_np[train_idx].shape)
    train_generated_confidence = scatter(train_metadata, generated_train[1], observations_np[train_idx].shape)
    val_real_evidence, val_real_confidence = ordinary(val_embedding, val_metadata, train_embedding, train_labels, observations_np[val_idx].shape)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    test_retrieval_started = time.perf_counter()
    test_real_evidence, test_real_confidence = ordinary(test_embedding, test_metadata, train_embedding, train_labels, observations_np[test_idx].shape)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    test_retrieval_seconds = time.perf_counter() - test_retrieval_started
    val_generated_evidence, val_generated_confidence = ordinary(val_embedding, val_metadata, generated, generated_labels, observations_np[val_idx].shape)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    test_retrieval_started = time.perf_counter()
    test_generated_evidence, test_generated_confidence = ordinary(test_embedding, test_metadata, generated, generated_labels, observations_np[test_idx].shape)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    test_retrieval_seconds += time.perf_counter() - test_retrieval_started
    selection = memory_lib.select_fused_weight(sources_np[val_idx], observations_np[val_idx], q_direct[val_idx], val_real_evidence, val_real_confidence, val_generated_evidence, val_generated_confidence, args.memory_weight_grid, args.generation_mix_grid)
    values = {'real_mean': selection['real_mean'], 'real_scale': selection['real_scale'], 'generated_mean': selection['generated_mean'], 'generated_scale': selection['generated_scale'], 'generation_mix': selection['generation_mix'], 'weight': selection['weight']}
    q_memory = q_direct.copy()
    for indices, real_evidence, real_confidence, generated_evidence, generated_confidence in ((train_idx, train_real_evidence, train_real_confidence, train_generated_evidence, train_generated_confidence), (val_idx, val_real_evidence, val_real_confidence, val_generated_evidence, val_generated_confidence), (test_idx, test_real_evidence, test_real_confidence, test_generated_evidence, test_generated_confidence)):
        q_memory[indices] = memory_lib.fused_probability(q_direct[indices], real_evidence, real_confidence, generated_evidence, generated_confidence, observations_np[indices], **values)
    old_predictions = memory_root / 'vae_memory' / 'test_predictions.npz'
    reconstruction_error = math.nan
    if old_predictions.exists():
        with np.load(old_predictions) as saved:
            reconstruction_error = float(np.max(np.abs(saved['y_score'] - q_memory[test_idx])))
        if reconstruction_error > args.memory_reconstruction_tolerance:
            raise RuntimeError(f'VAE memory mismatch {scenario} seed={seed}: {reconstruction_error:.3e}')
    audit = {**generation_audit, 'generation_seconds': generation_seconds, 'vae_final_loss': history[-1]['loss'], 'reconstruction_max_abs_error': reconstruction_error, 'memory_weight': selection['weight'], 'generation_mix': selection['generation_mix'], 'train_memory_episode_loo': True, 'test_memory_retrieval_seconds': test_retrieval_seconds}
    del encoder, vae
    return (q_memory, audit)

def train_source_model(graph: base.GraphOperator, sources: torch.Tensor, observations: torch.Tensor, q_scores: torch.Tensor, train_idx: np.ndarray, val_idx: np.ndarray, degree: torch.Tensor, clustering: torch.Tensor, args: argparse.Namespace, seed: int, output: Path, initialization_seed: int) -> tuple[source_lib.ConditionalSourceVAE, dict[str, Any]]:
    base.seed_everything(initialization_seed)
    model = source_lib.ConditionalSourceVAE(graph, args.hidden, args.layers, args.source_latent_dim, args.dropout).to(sources.device)
    checkpoint = output / 'source_vae.pt'
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=sources.device, weights_only=False))
        best = json.loads((output / 'source_vae_best.json').read_text())
    else:
        source_args = copy.copy(args)
        source_args.early_stopping_checks = args.source_early_stopping_checks
        history, best = source_lib.train_source_vae(model, sources[train_idx], observations[train_idx], q_scores[train_idx], sources[val_idx], observations[val_idx], q_scores[val_idx], degree, clustering, source_args, seed + 901)
        torch.save(model.state_dict(), checkpoint)
        write_json(output / 'source_vae_history.json', history)
        write_json(output / 'source_vae_best.json', best)
    model.eval()
    return (model, best)

def evaluate(rows: list[dict[str, Any]], graph_nx: Any, labels: np.ndarray, val_labels: np.ndarray, val_scores: np.ndarray, test_scores: np.ndarray, common: dict[str, Any], **extra: Any) -> None:
    _, _, threshold = source_lib.source_validation(val_labels, val_scores)
    val_metrics = source_lib.evaluate_scores(graph_nx, val_labels, val_scores, threshold, skip_aed=True)
    metrics = source_lib.evaluate_scores(graph_nx, labels, test_scores, threshold, skip_aed=True)
    row = {**common, 'model': 'MAGI', **extra, **{f'val_{key}': value for key, value in val_metrics.items()}, **{f'test_{key}': value for key, value in metrics.items()}}
    rows.append(row)
    print(f"[result] {common['dataset']}_{common['mechanism']} seed={common['seed']} F1={row['test_f1']:.4f} AUC={row['test_auc']:.4f}", flush=True)


def load_source_model(graph: base.GraphOperator, checkpoint: Path, args: argparse.Namespace, device: torch.device) -> source_lib.ConditionalSourceVAE:
    model = source_lib.ConditionalSourceVAE(graph, args.hidden, args.layers, args.source_latent_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    return model


def run() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    use_cuda = args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
    device = torch.device('cuda' if use_cuda else 'cpu')
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / 'run_config.json', vars(args))
    metrics_path = args.output_root / 'metrics.csv'
    rows = load_rows(metrics_path)
    completed = {(row['dataset'], row['mechanism'], int(row['seed'])) for row in rows}
    for dataset in args.datasets:
        for mechanism in args.mechanisms:
            scenario = f'{dataset}_{mechanism}'
            graph_nx, sources_np, observations_np, audit = base.load_baseline_scenario(dataset, mechanism, args.data_root, args.expected_samples)
            adjacency, degree_np, clustering_np, _ = base.graph_features(graph_nx)
            train_idx, val_idx, test_idx = base.baseline_split(len(sources_np), args.split_seed)
            graph = base.GraphOperator(adjacency).to(device)
            tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
            sources = tensor(sources_np)
            observations = tensor(observations_np)
            degree = tensor(degree_np)
            clustering = tensor(clustering_np)
            for seed in args.seeds:
                key = (dataset, mechanism, seed)
                if key in completed:
                    print(f'[skip] {scenario} seed={seed}', flush=True)
                    continue
                print(f'[run] {scenario} seed={seed} mode={args.mode}', flush=True)
                started = time.perf_counter()
                output = args.output_root / scenario / f'seed{seed}'
                output.mkdir(parents=True, exist_ok=True)
                final_root = args.checkpoint_root / 'final' / scenario / f'seed{seed}'
                if args.mode == 'evaluate':
                    with np.load(final_root / 'memory_conditioned_q.npz') as saved:
                        q_scores = saved['q_memory']
                    source_model = load_source_model(graph, final_root / 'source_vae' / 'source_vae.pt', args, device)
                    memory_audit = json.loads((final_root / 'memory_audit.json').read_text(encoding='utf-8'))
                else:
                    proposal_checkpoint = memory_lib.find_proposal([args.checkpoint_root / 'proposal'], scenario, seed)
                    q_direct = forward_lib.make_rag_proposal_scores(graph, adjacency, degree_np, clustering_np, sources_np, observations_np, train_idx, val_idx, test_idx, proposal_checkpoint, device, args.dropout, args.rag_topk)
                    memory_root = args.checkpoint_root / 'memory' / scenario / f'seed{seed}'
                    q_scores, memory_audit = build_memory_conditioned_q(scenario, seed, memory_root, adjacency, degree_np, clustering_np, sources_np, observations_np, train_idx, val_idx, test_idx, q_direct, args, device)
                    np.savez_compressed(output / 'memory_conditioned_q.npz', q_memory=q_scores)
                    write_json(output / 'memory_audit.json', memory_audit)
                    source_root = output / 'source_vae'
                    source_root.mkdir(exist_ok=True)
                    source_model, _ = train_source_model(graph, sources, observations, tensor(q_scores), train_idx, val_idx, degree, clustering, args, seed, source_root, seed + 800)
                val_particles = forward_lib.source_particles(source_model, q_scores[val_idx], observations_np[val_idx], degree, clustering, args.particles, seed + 1000, device)
                test_started = time.perf_counter()
                test_particles = forward_lib.source_particles(source_model, q_scores[test_idx], observations_np[test_idx], degree, clustering, args.particles, seed + 1100, device)
                del source_model
                forward_model = forward_lib.InvertibleResidualCVAE(graph, args.hidden, args.layers, args.source_latent_dim, args.dropout, args.mean_loss_weight, args.residual_penalty).to(device)
                forward_checkpoint = args.checkpoint_root / 'forward' / scenario / f'seed{seed}' / 'forward.pt'
                forward_model.load_state_dict(torch.load(forward_checkpoint, map_location=device, weights_only=False))
                forward_model.eval()
                val_q, val_forward, _ = joint_lib.particle_energies(val_particles, q_scores[val_idx], observations_np[val_idx], forward_model, degree, clustering, args.replay_draws, seed + 5000)
                test_q, test_forward, forward_audit = joint_lib.particle_energies(test_particles, q_scores[test_idx], observations_np[test_idx], forward_model, degree, clustering, args.replay_draws, seed + 6000)
                val_score, _ = joint_lib.aggregate_particles(val_particles, val_q, val_forward, args.forward_weight, args.weight_temperature)
                test_score, effective_sample_size = joint_lib.aggregate_particles(test_particles, test_q, test_forward, args.forward_weight, args.weight_temperature)
                inference_seconds = time.perf_counter() - test_started
                common = {'dataset': dataset, 'mechanism': mechanism, 'seed': seed, 'input_sha256': audit['sha256'], 'test_samples': len(test_idx), 'known_k_at_inference': False, 'intermediate_frames_used': False}
                evaluate(rows, graph_nx, sources_np[test_idx], sources_np[val_idx], val_score, test_score, common, particle_ess=effective_sample_size, forward_replay_variance=forward_audit['forward_replay_variance'], memory_weight=memory_audit['memory_weight'], memory_generation_mix=memory_audit['generation_mix'], inference_seconds=inference_seconds, total_seconds=time.perf_counter() - started)
                np.savez_compressed(output / 'test_predictions.npz', y_true=sources_np[test_idx], y_score=test_score)
                write_rows(metrics_path, rows)
                completed.add(key)
                del forward_model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            del graph
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    print(f'[saved] {metrics_path}', flush=True)
