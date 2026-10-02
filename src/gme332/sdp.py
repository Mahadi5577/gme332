"""Numerical SDPs (CVXPY + CLARABEL): PPT-mixture threshold, decomposable
witness value, genuine multipartite negativity, symmetry-reduced family SDP."""
import itertools
from typing import Dict, Tuple

import cvxpy as cp
import numpy as np

from .config import CUTS, SOLVER
from .states import ghz_family_ket


def _pt(X, dims, parties):
    for q in parties:
        X = cp.partial_transpose(X, dims, q)
    return X


def _var(D, real):
    return cp.Variable((D, D), symmetric=True) if real else cp.Variable((D, D), hermitian=True)


def ppt_mixture_noise_threshold(rho, dims, cuts=CUTS, real=False) -> float:
    """Smallest white-noise weight q such that (1-q) rho + q 1/D is a PPT mixture."""
    D = rho.shape[0]
    q = cp.Variable(); parts = [_var(D, real) for _ in cuts]
    cons = [(1 - q) * rho + q * np.eye(D) / D == sum(parts)]
    for X, k in zip(parts, cuts.values()):
        cons += [X >> 0, _pt(X, dims, k) >> 0]
    cp.Problem(cp.Minimize(q), cons).solve(solver=SOLVER)
    return float(q.value)


def decomposable_witness_value(rho, dims, cuts=CUTS, real=False) -> Tuple[float, np.ndarray]:
    """min Tr(W rho) over fully decomposable W (W = P_k + T_k(Q_k), P_k, Q_k >= 0)
    with Tr W = 1.  Negative  <=>  rho is not a PPT mixture  =>  rho is GME."""
    D = rho.shape[0]
    W = _var(D, real); cons = [cp.real(cp.trace(W)) == 1]
    for k in cuts.values():
        P, Q = _var(D, real), _var(D, real)
        cons += [P >> 0, Q >> 0, W == P + _pt(Q, dims, k)]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(W @ rho))), cons)
    prob.solve(solver=SOLVER)
    return float(prob.value), W.value


def genuine_multipartite_negativity(rho, dims, cuts=CUTS, real=False) -> float:
    """N_g(rho) = -min Tr(W rho), W = P_k + T_k(Q_k), 0 <= P_k, Q_k <= 1
    [Jungnitsch, Moroder, Guehne 2011].  N_g > 0  <=>  not a PPT mixture."""
    D = rho.shape[0]; I = np.eye(D)
    W = _var(D, real); cons = []
    for k in cuts.values():
        P, Q = _var(D, real), _var(D, real)
        cons += [P >> 0, Q >> 0, P << I, Q << I, W == P + _pt(Q, dims, k)]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(W @ rho))), cons)
    prob.solve(solver=SOLVER)
    return max(0.0, -float(prob.value))


def _torus_charge(a, b, c, d):
    """Charge of |abc> under local diagonal unitaries that fix |psi_d>
    (free phases: alpha, beta_0..beta_{d-1}, gamma_1)."""
    v = np.zeros(d + 2, dtype=int)
    v[0] += a; v[1 + b] += 1
    if c == 0:   v[0] += 1; v[1] -= 1; v[2] += 1; v[-1] += 1
    elif c == 1: v[-1] += 1
    else:        v[2] += 1; v[1 + c] -= 1; v[-1] += 1
    return tuple(v)


def family_threshold_sdp(d: int) -> float:
    """PPT-mixture threshold for |psi_d>, reduced to real torus-invariant
    operators (twirling a feasible point keeps it feasible)."""
    dims = [2, d, d]; D = 2 * d * d
    basis = list(itertools.product(range(2), range(d), range(d)))
    ch = [_torus_charge(*s, d) for s in basis]
    rho = np.outer(ghz_family_ket(d), ghz_family_ket(d))
    q = cp.Variable(); parts = []; cons = []
    for k in CUTS.values():
        X = cp.Variable((D, D), symmetric=True)
        cons += [X[i, j] == 0 for i in range(D) for j in range(i, D) if ch[i] != ch[j]]
        cons += [X >> 0, _pt(X, dims, k) >> 0]; parts.append(X)
    cons += [(1 - q) * rho + q * np.eye(D) / D == sum(parts)]
    cp.Problem(cp.Minimize(q), cons).solve(solver=SOLVER)
    return float(q.value)


def weak_constraint_pitfall(rho, dims) -> Dict:
    """Solve min Tr(W rho) s.t. Tr W = 1, W + T_k(W) >= 0 (a constraint that is
    NOT sufficient for a witness) and evaluate the optimiser on the product state
    (|0>+i|1>)/sqrt2 (x) (|00> - i|11> - i|22>)/sqrt3."""
    D = rho.shape[0]
    W = cp.Variable((D, D), symmetric=True)
    cons = [cp.trace(W) == 1] + [W + _pt(W, dims, k) >> 0 for k in CUTS.values()]
    prob = cp.Problem(cp.Minimize(cp.trace(W @ rho)), cons); prob.solve(solver=SOLVER)
    a   = np.array([1, 1j]) / np.sqrt(2)
    phi = np.zeros(9, complex); phi[0] = 1; phi[4] = -1j; phi[8] = -1j; phi /= np.sqrt(3)
    sig = np.kron(a, phi)
    return dict(value=float(prob.value),
                naive_threshold=abs(prob.value) / (abs(prob.value) + 1 / D),
                value_on_product_state=float(np.real(sig.conj() @ W.value @ sig)))
