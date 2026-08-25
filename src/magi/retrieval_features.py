from __future__ import annotations
import numpy as np
from . import graph_data as base

def _local_features(observations: np.ndarray, adjacency: np.ndarray, degree: np.ndarray) -> np.ndarray:
    denominator = np.maximum(base.adjacency_row_sum(adjacency), 1.0)
    one_hop = base.propagate_features(observations, adjacency) / denominator[None, :]
    two_hop = base.propagate_features(one_hop, adjacency) / denominator[None, :]
    three_hop = base.propagate_features(two_hop, adjacency) / denominator[None, :]
    boundary = np.abs(observations - one_hop)
    infection_ratio = np.broadcast_to(observations.mean(axis=1, keepdims=True), observations.shape)
    return np.stack([observations, one_hop, two_hop, three_hop, boundary, np.broadcast_to(degree, observations.shape), infection_ratio], axis=-1).astype(np.float32)

def _global_features(local: np.ndarray, observations: np.ndarray) -> np.ndarray:
    mean = local.mean(axis=1)
    std = local.std(axis=1)
    infected_count = observations.sum(axis=1, keepdims=True).clip(min=1.0)
    infected_mean = (local * observations[:, :, None]).sum(axis=1) / infected_count
    values = np.concatenate([mean, std, infected_mean], axis=-1)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-06)

def build_local_memory(train_observations: np.ndarray, train_sources: np.ndarray, query_observations: np.ndarray, adjacency: np.ndarray, degree: np.ndarray, topk: int) -> np.ndarray:
    train_local = _local_features(train_observations, adjacency, degree)
    query_local = _local_features(query_observations, adjacency, degree)
    train_norm = train_local / np.maximum(np.linalg.norm(train_local, axis=-1, keepdims=True), 1e-06)
    query_norm = query_local / np.maximum(np.linalg.norm(query_local, axis=-1, keepdims=True), 1e-06)
    train_global = _global_features(train_local, train_observations)
    query_global = _global_features(query_local, query_observations)
    leave_one_out = len(train_observations) == len(query_observations) and np.array_equal(train_observations, query_observations)
    available = len(train_observations) - int(leave_one_out)
    topk = max(1, min(int(topk), available))
    memory = np.zeros((len(query_observations), train_sources.shape[1]), dtype=np.float32)
    node_indices = np.arange(train_sources.shape[1])[None, :]
    for index, query in enumerate(query_norm):
        local_similarity = np.einsum('nf,tnf->tn', query, train_norm)
        global_similarity = query_global[index] @ train_global.T
        similarity = 0.75 * local_similarity + 0.25 * global_similarity[:, None]
        if leave_one_out:
            similarity[index, :] = -np.inf
        top = np.argpartition(similarity, -topk, axis=0)[-topk:]
        selected_similarity = np.take_along_axis(similarity, top, axis=0)
        weights = np.exp((selected_similarity - selected_similarity.max(axis=0, keepdims=True)) / 0.1)
        weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-08)
        selected_sources = train_sources[top, node_indices]
        memory[index] = np.sum(weights * selected_sources, axis=0)
    return memory
