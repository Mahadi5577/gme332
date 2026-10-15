"""Numerical SDPs (CVXPY + CLARABEL): PPT-mixture threshold, decomposable
witness value, genuine multipartite negativity, symmetry-reduced family SDPs
(primal and dual)."""
import itertools
import sys
import warnings
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np

from .config import CUTS, SOLVER
from .states import ghz_family_ket


class SolverError(RuntimeError):
    """Raised when an SDP does not reach an optimal status."""


def _solve(prob: cp.Problem) -> float:
    """Solve with the configured solver and check the status: 'optimal' is
    accepted, 'optimal_inaccurate' is accepted with a RuntimeWarning, anything
    else raises SolverError."""
    caller = sys._getframe(1).f_code.co_name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)     # CVXPY's own message; reported below with context
        prob.solve(solver=SOLVER)
    if prob.status == cp.OPTIMAL:
        return float(prob.value)
    if prob.status == cp.OPTIMAL_INACCURATE:
        warnings.warn(f"{caller}: SDP solved only to reduced accuracy (status {prob.status}, "
                      f"value {prob.value:.3e})", RuntimeWarning, stacklevel=3)
        return float(prob.value)
    raise SolverError(f"{caller}: SDP failed with status {prob.status!r}")


def _pt(X, dims, parties):
    for q in parties:
        X = cp.partial_transpose(X, dims, q)
    return X


def _var(D, real):
    return cp.Variable((D, D), symmetric=True) if real else cp.Variable((D, D), hermitian=True)


def _objective(W, rho, real):
    """Tr(W rho) as a real affine expression. With real=True the state must be
    real (up to rounding); its real part is used, so real- and complex-dtype
    inputs are both accepted."""
    rho = np.asarray(rho)
    if real:
        if np.iscomplexobj(rho) and np.abs(rho.imag).max() > 1e-12:
            raise ValueError("real=True requires a real density matrix")
        return cp.trace(W @ np.real(rho))
    return cp.real(cp.trace(W @ rho))


def ppt_mixture_noise_threshold(rho, dims, cuts=CUTS, real=False) -> float:
    """Smallest white-noise weight q such that (1-q) rho + q 1/D is a PPT mixture."""
    D = rho.shape[0]
    if real:
        rho = np.real(rho)
    q = cp.Variable(); parts = [_var(D, real) for _ in cuts]
    cons = [(1 - q) * rho + q * np.eye(D) / D == sum(parts)]
    for X, k in zip(parts, cuts.values()):
        cons += [X >> 0, _pt(X, dims, k) >> 0]
    return _solve(cp.Problem(cp.Minimize(q), cons))


def decomposable_witness_value(rho, dims, cuts=CUTS, real=False) -> Tuple[float, np.ndarray]:
    """min Tr(W rho) over fully decomposable W (W = P_k + T_k(Q_k), P_k, Q_k >= 0)
    with Tr W = 1.  Negative  <=>  rho is not a PPT mixture  =>  rho is GME."""
    D = rho.shape[0]
    W = _var(D, real); cons = [(cp.trace(W) if real else cp.real(cp.trace(W))) == 1]
    for k in cuts.values():
        P, Q = _var(D, real), _var(D, real)
        cons += [P >> 0, Q >> 0, W == P + _pt(Q, dims, k)]
    value = _solve(cp.Problem(cp.Minimize(_objective(W, rho, real)), cons))
    return value, W.value


def genuine_multipartite_negativity(rho, dims, cuts=CUTS, real=False) -> float:
    """N_g(rho) = -min Tr(W rho), W = P_k + T_k(Q_k), 0 <= P_k, Q_k <= 1
    [Jungnitsch, Moroder, Guehne 2011].  N_g > 0  <=>  not a PPT mixture.
    Small negative optimal values (solver noise around zero) are reported as 0."""
    D = rho.shape[0]; I = np.eye(D)
    W = _var(D, real); cons = []
    for k in cuts.values():
        P, Q = _var(D, real), _var(D, real)
        cons += [P >> 0, Q >> 0, P << I, Q << I, W == P + _pt(Q, dims, k)]
    value = _solve(cp.Problem(cp.Minimize(_objective(W, rho, real)), cons))
    if value > 1e-7:
        warnings.warn(f"positive optimal value {value:.2e} in N_g (should be <= 0)", RuntimeWarning)
    return max(0.0, -value)


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
    return _solve(cp.Problem(cp.Minimize(q), cons))


# ---------------------------------------------------------------------------
# Block-reduced dual SDP for the 2 x d x d family
# ---------------------------------------------------------------------------

def _blocks(mask: np.ndarray) -> List[List[int]]:
    """Connected components of the sparsity graph of a D x D boolean mask."""
    D = mask.shape[0]; adj = mask | mask.T; seen = np.zeros(D, bool); out = []
    for s in range(D):
        if seen[s]:
            continue
        stack, comp = [s], []
        while stack:
            k = stack.pop()
            if seen[k]:
                continue
            seen[k] = True; comp.append(k); stack += list(np.nonzero(adj[k])[0])
        out.append(sorted(comp))
    return out


def _pt_permutation(basis, parties) -> np.ndarray:
    """perm such that vec(T(X)) = vec(X)[perm] (row-major vec, D x D)."""
    flat = {s: i for i, s in enumerate(basis)}; D = len(basis)
    perm = np.zeros(D * D, dtype=int)
    for i, s in enumerate(basis):
        for j, t in enumerate(basis):
            a, b = list(s), list(t)
            for p in parties:
                a[p], b[p] = b[p], a[p]
            perm[flat[tuple(a)] * D + flat[tuple(b)]] = i * D + j
    return perm


def family_witness_sdp(d: int) -> Dict:
    """Dual of family_threshold_sdp: min Tr(W rho_d) over fully decomposable W
    with Tr W = 1, reduced by symmetry.  Twirling over the torus keeps P_k and
    R_k := T_k(Q_k) invariant, so both are parametrised in the invariant
    pattern, Q_k = T_k(R_k) >= 0 is imposed, and every PSD constraint is split
    into the connected blocks of its sparsity pattern (block size <= d).
    Returns c_min and the threshold |c|/(|c| + 1/D) it implies."""
    D = 2 * d * d
    basis = list(itertools.product(range(2), range(d), range(d)))
    ch = [_torus_charge(*s, d) for s in basis]
    allowed = [(i, j) for i in range(D) for j in range(i, D) if ch[i] == ch[j]]
    B = np.zeros((D * D, len(allowed)))
    for k, (i, j) in enumerate(allowed):
        B[i * D + j, k] = 1; B[j * D + i, k] = 1
    mask = (B.sum(axis=1) > 0).reshape(D, D)

    def psd(vec, pattern):
        M = cp.reshape(vec, (D, D), order="C")
        return [M[np.ix_(c, c)] >> 0 if len(c) > 1 else M[c[0], c[0]] >= 0 for c in _blocks(pattern)]

    cons, W = [], None
    for parties in CUTS.values():
        perm = _pt_permutation(basis, parties)
        Pv = B @ cp.Variable(len(allowed)); Rv = B @ cp.Variable(len(allowed))
        cons += psd(Pv, mask) + psd(Rv[perm], mask.reshape(-1)[perm].reshape(D, D))
        if W is None:
            W = Pv + Rv
        else:
            cons.append(Pv + Rv == W)
    rho = np.outer(ghz_family_ket(d), ghz_family_ket(d)).reshape(-1)
    cons.append(cp.trace(cp.reshape(W, (D, D), order="C")) == 1)
    c = _solve(cp.Problem(cp.Minimize(rho @ W), cons))
    return dict(c_min=c, threshold=abs(c) / (abs(c) + 1.0 / D))


def weak_constraint_pitfall(rho, dims) -> Dict:
    """Solve min Tr(W rho) s.t. Tr W = 1, W + T_k(W) >= 0 (a constraint that is
    NOT sufficient for a witness) and evaluate the optimiser on the product state
    (|0>+i|1>)/sqrt2 (x) (|00> - i|11> - i|22>)/sqrt3."""
    D = rho.shape[0]
    W = cp.Variable((D, D), symmetric=True)
    cons = [cp.trace(W) == 1] + [W + _pt(W, dims, k) >> 0 for k in CUTS.values()]
    value = _solve(cp.Problem(cp.Minimize(cp.trace(W @ np.real(rho))), cons))
    a   = np.array([1, 1j]) / np.sqrt(2)
    phi = np.zeros(9, complex); phi[0] = 1; phi[4] = -1j; phi[8] = -1j; phi /= np.sqrt(3)
    sig = np.kron(a, phi)
    return dict(value=value,
                naive_threshold=abs(value) / (abs(value) + 1 / D),
                value_on_product_state=float(np.real(sig.conj() @ W.value @ sig)))
