"""States, partial transpose and negativity (numerical, NumPy)."""
from typing import Sequence

import numpy as np

from .basis import flat_index


def ghz_family_ket(d: int) -> np.ndarray:
    """|psi_d> = (|000> + sum_{j=1}^{d-1} |1jj>) / sqrt(d)  in C^2 (x) C^d (x) C^d."""
    dims = [2, d, d]
    ket  = np.zeros(2 * d * d)
    ket[flat_index((0, 0, 0), dims)] = 1.0
    for j in range(1, d):
        ket[flat_index((1, j, j), dims)] = 1.0
    return ket / np.linalg.norm(ket)


def isotropic_noise(rho_pure: np.ndarray, p: float) -> np.ndarray:
    D = rho_pure.shape[0]
    return (1 - p) * rho_pure + p * np.eye(D) / D


def partial_transpose(rho: np.ndarray, dims: Sequence[int],
                      parties: Sequence[int]) -> np.ndarray:
    """Reshape to (row_0..row_{N-1}, col_0..col_{N-1}) and swap the row/col
    axes of every transposed party (valid for unequal local dimensions)."""
    dims = list(dims); N = len(dims); D = int(np.prod(dims))
    t = rho.reshape(dims + dims)
    for p in parties:
        t = np.swapaxes(t, p, p + N)
    return t.reshape(D, D)


def negativity(rho: np.ndarray, dims, parties) -> float:
    ev = np.linalg.eigvalsh(partial_transpose(rho, dims, parties))
    return float(-ev[ev < 0].sum())


def negativity_right_derivative(rho, drho, dims, parties, tol=1e-9) -> float:
    """One-sided derivative d/dt N(rho + t drho) at t = 0+.
    Negative eigenspace contributes -Tr(Pi_- X'), and eigenvalues in the kernel
    of T(rho) that are pushed below zero contribute |mu| (degenerate
    first-order perturbation theory on Pi_0 X' Pi_0)."""
    X  = partial_transpose(rho, dims, parties)
    dX = partial_transpose(drho, dims, parties)
    ev, V = np.linalg.eigh(X)
    Vn = V[:, ev < -tol]; V0 = V[:, np.abs(ev) <= tol]
    rate = -float(np.real(np.trace(Vn.conj().T @ dX @ Vn)))
    if V0.shape[1]:
        mu = np.linalg.eigvalsh(V0.conj().T @ dX @ V0)
        rate += -float(mu[mu < 0].sum())
    return rate
