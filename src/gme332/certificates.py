"""Exact certificates: the fully decomposable witness family W_d and the
exact PPT-mixture decomposition of rho(p*) at d = 3."""
from fractions import Fraction
from typing import Dict

from .basis import flat_index
from .config import CUTS
from .exact import ExMat, QSqrt, ex_is_psd, ex_partial_transpose, ex_swap_BC


def exact_witness_family(d: int) -> Dict:
    """Unnormalised fully decomposable witness W~_d for |psi_d> and, for every
    cut k, its decomposition W~_d = P_k + T_k(Q_k) with P_k, Q_k >= 0.
    lam = 1/sqrt(d-1); all entries lie in Q(sqrt(d-1))."""
    m    = d - 1
    dims = [2, d, d]; D = 2 * d * d
    ix   = lambda a, b, c: flat_index((a, b, c), dims)
    one  = QSqrt(1, 0, m)
    rt   = QSqrt(0, 1, m)                       # sqrt(m)
    lam  = QSqrt(0, Fraction(1, m), m)          # 1/sqrt(m)
    js   = range(1, d)
    ghz0 = {ix(0, j, j): one for j in js}       # sum_j |0jj>

    W = ExMat(D, m)
    for j in js:
        W.add_sym(ix(0, 0, 0), ix(1, j, j), -one)
        for s in (ix(0, 0, j), ix(0, j, 0), ix(1, 0, j), ix(1, j, 0)):
            W.add(s, s, one)
    W.add_outer(ghz0, lam)
    W.add(ix(1, 0, 0), ix(1, 0, 0), rt)

    # cut A|BC
    P1 = ExMat(D, m)
    for j in js:
        for s in (ix(0, 0, j), ix(0, j, 0), ix(1, 0, j), ix(1, j, 0)):
            P1.add(s, s, one)
    u = dict(ghz0); u[ix(1, 0, 0)] = -rt
    Q1 = ExMat(D, m).add_outer(u, lam)
    # cut AB|C  (and AC|B by B<->C symmetry)
    def cut_pair(pairs, diag):
        P = ExMat(D, m).add_outer(ghz0, lam)
        P.add(ix(1, 0, 0), ix(1, 0, 0), rt)
        for s in diag: P.add(s, s, one)
        Q = ExMat(D, m)
        for a, b in pairs: Q.add_outer({a: one, b: -one})
        return P, Q
    P2, Q2 = cut_pair([(ix(0, 0, j), ix(1, j, 0)) for j in js],
                      [s for j in js for s in (ix(0, j, 0), ix(1, 0, j))])
    P3, Q3 = cut_pair([(ix(0, j, 0), ix(1, 0, j)) for j in js],
                      [s for j in js for s in (ix(0, 0, j), ix(1, j, 0))])
    return dict(d=d, m=m, dims=dims, D=D, W=W,
                decomposition={"A|BC": (P1, Q1), "AB|C": (P2, Q2), "AC|B": (P3, Q3)})


def verify_witness_family(d: int) -> Dict:
    """Exact checks: W~ = P_k + T_k(Q_k), P_k, Q_k >= 0 for all k; Tr W~; <psi~|W~|psi~>;
    threshold p*(d) at which Tr(W~ rho(p)) = 0."""
    wf = exact_witness_family(d)
    W, dims, D, m = wf["W"], wf["dims"], wf["D"], wf["m"]
    ok_decomp, ok_psd = True, True
    for name, (P, Q) in wf["decomposition"].items():
        ok_decomp &= (P + ex_partial_transpose(Q, dims, CUTS[name])).equals(W)
        ok_psd    &= ex_is_psd(P) and ex_is_psd(Q)
    psi_t = ExMat(D, m).add_outer({flat_index((0, 0, 0), dims): QSqrt(1, 0, m),
                                   **{flat_index((1, j, j), dims): QSqrt(1, 0, m)
                                      for j in range(1, d)}})
    trW      = W.trace()                          # 4(d-1) + 2 sqrt(d-1)
    wpsi     = W.inner(psi_t) / d                 # Tr(W~ rho_psi) = -2(d-1)/d
    p_star   = (-wpsi) / ((-wpsi) + trW / D)      # root of (1-p) wpsi + p trW / D
    return dict(d=d, decomposition_exact=ok_decomp, P_Q_psd_exact=ok_psd,
                trace_W=trW, tr_W_rho_psi=wpsi, c_normalised=wpsi / trW,
                p_star=p_star, W_numeric=W.to_numpy() / float(trW))


def exact_primal_certificate_d3() -> Dict:
    """Exact PPT-mixture decomposition rho(p*) = rho_1 + rho_2 + rho_3 at
    p* = 12/(16+sqrt2), rho_k >= 0 and T_k(rho_k) >= 0.  rho_2 is listed
    explicitly, rho_3 = S rho_2 S (S swaps B and C), rho_1 = rho(p*) - rho_2 - rho_3.
    Entries lie in Q(sqrt2); n = p*/18 = (16 - sqrt2)/381."""
    d, dims, D = 3, [2, 3, 3], 18
    ix  = lambda s: flat_index(tuple(int(ch) for ch in s), dims)
    R   = lambda num, den: QSqrt(Fraction(num, den), 0, 2)
    p_star = QSqrt(12, 0, 2) / QSqrt(16, 1, 2)
    n   = p_star / 18
    mm  = (1 - p_star) / 3
    rho_p = ExMat(D, 2)
    for k in range(D): rho_p.add(k, k, n)
    for a in ("000", "111", "122"):
        for b in ("000", "111", "122"): rho_p.add(ix(a), ix(b), mm)

    rho2 = ExMat(D, 2)
    entries = [("000", "000", R(13, 200)), ("000", "111", n), ("000", "122", n),
               ("001", "001", n), ("002", "002", n), ("110", "110", n), ("120", "120", n),
               ("011", "011", R(7, 1000)), ("022", "022", R(7, 1000)),
               ("011", "022", R(-7, 1000)),
               ("012", "012", R(13, 1000)), ("021", "021", R(13, 1000)),
               ("111", "111", R(31, 1000)), ("122", "122", R(31, 1000)),
               ("111", "122", R(3, 200)),
               ("112", "112", R(2, 125)), ("121", "121", R(2, 125))]
    for a, b, v in entries: rho2.add_sym(ix(a), ix(b), v)
    rho3 = ex_swap_BC(rho2, d)
    rho1 = rho_p - rho2 - rho3
    parts = {"A|BC": rho1, "AB|C": rho2, "AC|B": rho3}
    checks = {}
    for name, rk in parts.items():
        checks[name] = dict(rho_psd=ex_is_psd(rk),
                            pt_psd=ex_is_psd(ex_partial_transpose(rk, dims, CUTS[name])))
    # complementary slackness with the witness decomposition (optimality pairing)
    wf = exact_witness_family(3)
    slack = {name: (P.inner(parts[name]),
                    Q.inner(ex_partial_transpose(parts[name], dims, CUTS[name])))
             for name, (P, Q) in wf["decomposition"].items()}
    slack_zero = all(a.is_zero() and b.is_zero() for a, b in slack.values())
    sums_ok = (rho1 + rho2 + rho3).equals(rho_p)
    return dict(p_star=p_star, checks=checks, sum_equals_rho=sums_ok,
                complementary_slackness=slack_zero,
                all_ok=sums_ok and all(c["rho_psd"] and c["pt_psd"] for c in checks.values()),
                rho_parts={k: v.to_numpy() for k, v in parts.items()})
