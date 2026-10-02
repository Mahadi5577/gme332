"""Lindblad dynamics: Liouvillian, exact propagator, ESD times, GME lifetime."""
from typing import List

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

from .sdp import decomposable_witness_value
from .states import partial_transpose


def _embed(op, party, dims):
    ops = [np.eye(d) for d in dims]; ops[party] = op
    out = ops[0]
    for o in ops[1:]: out = np.kron(out, o)
    return out


def _z_centred(d):
    z = np.arange(d, dtype=float) - (d - 1) / 2.0
    return np.diag(z / np.linalg.norm(z))


def collapse_operators(dims, gamma_1, gamma_phi, gamma_corr) -> List[np.ndarray]:
    ops = []
    for p, d in enumerate(dims):
        ops.append(np.sqrt(gamma_1) * _embed(np.diag(np.sqrt(np.arange(1, d)), 1), p, dims))
        ops.append(np.sqrt(gamma_phi) * _embed(_z_centred(d), p, dims))
    if gamma_corr > 0:
        ops.append(np.sqrt(gamma_corr) * np.kron(np.kron(np.eye(dims[0]), _z_centred(dims[1])),
                                                 _z_centred(dims[2])))
    return ops


def liouvillian(dims, gamma_1, gamma_phi, gamma_corr) -> np.ndarray:
    """Row-major superoperator: vec(C rho C^+) = (C (x) C*) vec(rho)."""
    D = int(np.prod(dims)); I = np.eye(D); L = np.zeros((D * D, D * D), complex)
    for C in collapse_operators(dims, gamma_1, gamma_phi, gamma_corr):
        CdC = C.conj().T @ C
        L += np.kron(C, C.conj()) - 0.5 * np.kron(CdC, I) - 0.5 * np.kron(I, CdC.T)
    return L


class Propagator:
    """Exact solution rho(t) = exp(L t) rho(0) via the eigendecomposition of L,
    falling back to expm if L is numerically defective."""

    def __init__(self, L, rho0):
        self.L, self.v0, self.D = L, rho0.reshape(-1), rho0.shape[0]
        w, V = np.linalg.eig(L)
        self.ok = np.linalg.cond(V) < 1e8
        if self.ok:
            self.w, self.V, self.c = w, V, np.linalg.solve(V, self.v0)

    def __call__(self, t):
        if self.ok:
            v = self.V @ (np.exp(self.w * t) * self.c)
        else:
            v = la.expm(self.L * t) @ self.v0
        r = v.reshape(self.D, self.D)
        return (r + r.conj().T) / 2


def esd_time(prop, dims, parties, t_max, n_grid=301) -> float:
    """Exact ESD time: first root of lambda_min(T_k rho(t)) = 0."""
    f = lambda t: np.linalg.eigvalsh(partial_transpose(prop(t), dims, parties))[0]
    ts = np.linspace(0, t_max, n_grid)
    vals = [f(t) for t in ts]
    for i in range(1, n_grid):
        if vals[i] >= 0 > vals[i - 1]:
            return brentq(f, ts[i - 1], ts[i], xtol=1e-10)
    return float("nan")


def gme_lifetime(prop, dims, t_hi, xtol=1e-4) -> float:
    """First time rho(t) becomes a PPT mixture (decomposable-witness value = 0).
    t_hi must be a time at which rho is already PPT across some cut."""
    f = lambda t: decomposable_witness_value(prop(t), dims, real=True)[0]
    return brentq(f, 0.0, t_hi, xtol=xtol)
