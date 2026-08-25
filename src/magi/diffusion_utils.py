from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

class GraphOperator(nn.Module):

    def __init__(self, adjacency: np.ndarray | sp.spmatrix) -> None:
        super().__init__()
        n = adjacency.shape[0]
        if sp.issparse(adjacency):
            augmented = adjacency.tocsr().astype(np.float32) + sp.eye(n, dtype=np.float32, format='csr')
            degree = np.asarray(augmented.sum(axis=1)).reshape(-1)
            coo = augmented.tocoo()
            rows, cols = (coo.row, coo.col)
            weights = coo.data / np.sqrt(degree[rows] * degree[cols])
        else:
            augmented = np.asarray(adjacency, dtype=np.float32) + np.eye(n, dtype=np.float32)
            degree = augmented.sum(axis=1)
            rows, cols = np.nonzero(augmented)
            weights = augmented[rows, cols] / np.sqrt(degree[rows] * degree[cols])
        self.n = n
        self.register_buffer('rows', torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer('cols', torch.as_tensor(cols, dtype=torch.long))
        self.register_buffer('weights', torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        squeezed = values.ndim == 2
        if squeezed:
            values = values.unsqueeze(-1)
        messages = values[:, self.rows] * self.weights[None, :, None]
        result = torch.zeros(values.shape[0], self.n, values.shape[-1], dtype=values.dtype, device=values.device)
        result.index_add_(1, self.cols, messages)
        return result.squeeze(-1) if squeezed else result
