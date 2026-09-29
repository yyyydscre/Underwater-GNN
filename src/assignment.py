"""Small NumPy implementation of rectangular linear-sum assignment."""
from __future__ import annotations

import numpy as np


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a minimum-cost one-to-one assignment in O(n^3).

    This is the Hungarian shortest-augmenting-path algorithm. Keeping it local
    avoids a SciPy binary dependency for a problem whose matrices are at most
    a few dozen light centers.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    transposed = rows > cols
    if transposed:
        cost = cost.T
        rows, cols = cost.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(cols + 1, dtype=np.float64)
    parent = np.zeros(cols + 1, dtype=np.int64)
    way = np.zeros(cols + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        parent[0] = row
        column0 = 0
        min_value = np.full(cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = parent[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, cols + 1):
                if used[column]:
                    continue
                value = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if value < min_value[column]:
                    min_value[column] = value
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(cols + 1):
                if used[column]:
                    u[parent[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if parent[column0] == 0:
                break
        while True:
            column1 = way[column0]
            parent[column0] = parent[column1]
            column0 = column1
            if column0 == 0:
                break
    assigned_rows = parent[1:] - 1
    assigned_cols = np.arange(cols, dtype=np.int64)
    valid = assigned_rows >= 0
    assigned_rows = assigned_rows[valid]
    assigned_cols = assigned_cols[valid]
    order = np.argsort(assigned_rows)
    assigned_rows, assigned_cols = assigned_rows[order], assigned_cols[order]
    if transposed:
        return assigned_cols, assigned_rows
    return assigned_rows, assigned_cols

