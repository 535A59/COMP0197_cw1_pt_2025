import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations_with_replacement


_combinations_cache = {}
def get_combinations(D, M):
    key = (D, M)
    if key in _combinations_cache:
        return _combinations_cache[key]

    combos = []
    for m in range(M + 1):
        for comb in combinations_with_replacement(range(D), m):
            combos.append(comb)
    _combinations_cache[key] = combos
    return combos

def polynomial_transform(X, M):
    N, D = X.shape
    combos = get_combinations(D, M)
    device = X.device
    batch_size = X.shape[0]
    features = torch.stack([
        torch.prod(X[:, comb], dim=1) if len(comb) > 0
        else torch.ones(batch_size, device=device)
        for comb in combos
    ], dim=1)

    return features

def logistic_fun(w, M, X):
    features = polynomial_transform(X, M)
    y_linear = features @ w             
    return torch.sigmoid(y_linear)

def generate_data(N, D, M):
    p = len(get_combinations(D, M))

    indices = torch.arange(p, 0, -1, dtype=torch.float32)
    w_true = ((-1) ** indices) * (torch.sqrt(indices) / p)

    X = torch.rand(N, D) * 10.0 - 5.0

    y_probs = logistic_fun(w_true, M, X)
    y_labels = (y_probs >= 0.5).int()

    y_noisy = y_probs + torch.normal(mean=0.0, std=1.0, size=y_probs.shape)
    t_labels = (y_noisy >= 0.5).int()

    return X, t_labels, y_labels


def evaluate_accuracy(X, y_labels, w, M):
    X_poly = polynomial_transform(X, M)
    y_linear = X_poly @ w
    y_pred = torch.sigmoid(y_linear)
    t_pred = (y_pred >= 0.5).int()
    error = torch.mean((t_pred == y_labels).float())
    return error.item()


def evaluate_noisy_accuracy(y_noisy, y_true):
    return torch.mean((y_noisy == y_true).float()).item()