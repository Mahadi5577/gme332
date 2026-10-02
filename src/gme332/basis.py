"""Basis indexing helpers (pure Python)."""
from typing import Sequence


def flat_index(s: Sequence[int], dims: Sequence[int]) -> int:
    idx = 0
    for k, dk in zip(s, dims):
        idx = idx * dk + k
    return idx
