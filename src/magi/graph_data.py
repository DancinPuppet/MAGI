from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from .diffusion_utils import GraphOperator

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_baseline_scenario(dataset: str, mechanism: str, data_root: Path, expected_samples: int | None) -> tuple[nx.Graph, np.ndarray, np.ndarray, dict[str, Any]]:
    path = data_root / f'{dataset}_{mechanism}.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as values:
        node_count = int(values['node_count'])
        edge_index = values['edge_index'].astype(np.int64, copy=False)
        sources = values['sources'].astype(np.float32)
        observations = values['observations'].astype(np.float32)
        source_seeds = values['source_seeds'].astype(np.int64).tolist()
        diffusion_seeds = values['diffusion_seeds'].astype(np.int64).tolist()
    if expected_samples is not None and len(sources) != expected_samples:
        raise ValueError(f'{path.name}: expected {expected_samples} samples, got {len(sources)}')
    if sources.shape != observations.shape or sources.shape[1] != node_count:
        raise ValueError(f'Invalid source or observation shape in {path.name}')
    if np.any(sources > observations):
        raise ValueError(f'Source support violation in {path.name}')
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(edge_index.T.tolist())
    audit = {'path': str(path), 'sha256': sha256(path), 'dataset': dataset, 'mechanism': mechanism, 'samples': len(sources), 'nodes': node_count, 'edges': graph.number_of_edges(), 'components': nx.number_connected_components(graph), 'source_seeds': source_seeds, 'diffusion_seeds': diffusion_seeds, 'final_ratios': [float(row.mean()) for row in observations]}
    return graph, sources, observations, audit

def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def baseline_split(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    train_end = int(count * 0.8)
    validation_end = int(count * 0.9)
    return (np.asarray(order[:train_end], dtype=np.int64), np.asarray(order[train_end:validation_end], dtype=np.int64), np.asarray(order[validation_end:], dtype=np.int64))

def graph_features(graph: nx.Graph) -> tuple[np.ndarray | sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    node_count = graph.number_of_nodes()
    force_sparse = os.environ.get('MAGI_FORCE_SPARSE', '0') == '1'
    if force_sparse or node_count >= 5000:
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=list(range(node_count)), format='csr', dtype=np.float32).tocsr()
    else:
        adjacency = nx.to_numpy_array(graph, nodelist=list(range(node_count)), dtype=np.float32)
    degrees = np.asarray([graph.degree(node) for node in range(node_count)], dtype=np.float32)
    degree = degrees / max(float(degrees.max()), 1.0)
    clustering = np.asarray([nx.clustering(graph, node) for node in range(node_count)], dtype=np.float32)
    components = np.zeros(node_count, dtype=np.float32)
    for component in nx.connected_components(graph):
        components[list(component)] = len(component) / max(node_count, 1)
    return (adjacency, degree, clustering, components)

def adjacency_row_sum(adjacency: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(adjacency):
        return np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return np.asarray(adjacency.sum(axis=1)).reshape(-1)

def propagate_features(values: np.ndarray, adjacency: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(adjacency):
        return np.asarray(adjacency.dot(values.T).T)
    return values @ adjacency.T

def batch_indices(count: int, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(count)
    for start in range(0, count, batch_size):
        yield order[start:start + batch_size]

def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best = (-float('inf'), 0.5)
    for threshold in np.linspace(0.05, 0.95, 37):
        prediction = scores >= threshold
        value = float(np.mean([f1_score(target, output, zero_division=0) for target, output in zip(labels, prediction)]))
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]

def aed(graph: nx.Graph, label: np.ndarray, score: np.ndarray, threshold: float) -> float:
    true_nodes = set(np.flatnonzero(label >= 0.5).tolist())
    predicted_nodes = set(np.flatnonzero(score >= threshold).tolist())
    n = graph.number_of_nodes()
    if not true_nodes or not predicted_nodes:
        return float(n)
    component = {}
    for index, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            component[node] = index

    def directed(left: set[int], right: set[int]) -> float:
        values = []
        for node in left:
            distances = nx.single_source_shortest_path_length(graph, node)
            candidates = [distances[target] for target in right if component.get(target) == component.get(node) and target in distances]
            values.append(min(candidates) if candidates else n)
        return float(np.mean(values))
    return 0.5 * (directed(true_nodes, predicted_nodes) + directed(predicted_nodes, true_nodes))

def evaluate(graph: nx.Graph, labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    values = {key: [] for key in ('accuracy', 'precision', 'recall', 'f1', 'aed')}
    skip_aed = os.environ.get('MAGI_SKIP_AED', '0') == '1' or graph.number_of_nodes() >= 10000
    for label, prediction, score in zip(labels, predictions, scores):
        values['accuracy'].append(accuracy_score(label, prediction))
        values['precision'].append(precision_score(label, prediction, zero_division=0))
        values['recall'].append(recall_score(label, prediction, zero_division=0))
        values['f1'].append(f1_score(label, prediction, zero_division=0))
        if not skip_aed:
            values['aed'].append(aed(graph, label, score, threshold))
    result = {key: float(np.mean(items)) if items else float('nan') for key, items in values.items()}
    try:
        result['auc'] = float(roc_auc_score(labels.reshape(-1), scores.reshape(-1)))
    except ValueError:
        result['auc'] = float('nan')
    result['threshold'] = float(threshold)
    result['samples'] = float(len(labels))
    return result
