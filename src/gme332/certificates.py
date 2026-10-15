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
                rho_parts={k: v.to_numpy() for k, v in parts.items()},
                parts_exact=parts)


def _sub(M: ExMat, keep) -> ExMat:
    """Principal submatrix of an exact matrix on the index list `keep`."""
    pos = {k: i for i, k in enumerate(keep)}
    out = ExMat(len(keep), M.m)
    for (i, j), v in M.e.items():
        if i in pos and j in pos:
            out.add(pos[i], pos[j], v)
    return out


def biseparability_certificate_d3() -> Dict:
    """Exact proof that rho(p*) is BISEPARABLE (so p* = 12/(16+sqrt2) is the true GME threshold).

    rho_1 (cut A|BC): all off-diagonal entries lie in C^2 (x) span{00,11,22}_BC; that 2x3 block is
        PSD and PPT, hence separable (Horodecki 1996); the rest is diagonal (product states).
    rho_2 (cut AB|C): no coherences between span{01,02}_AB (x) C^3 and span{00,11,12}_AB (x) C^3.
        The first block is 2x3, PSD and PPT -> separable.  The second has the maximally correlated
        form sum A_ij |x_i i><x_j j| + sum_{i!=j} B_ij |x_i j><x_i j| with A doubly nonnegative
        (= completely positive for 3x3, Maxfield-Minc) and A_ij <= B_ij, B_ji -> separable.
    rho_3 = S rho_2 S is separable across AC|B by symmetry."""
    dims = [2, 3, 3]
    ix = lambda s: flat_index(tuple(int(ch) for ch in s), dims)
    prim = exact_primal_certificate_d3()
    rho1, rho2 = prim["parts_exact"]["A|BC"], prim["parts_exact"]["AB|C"]
    labels = ["".join(map(str, (a, b, c))) for a in range(2) for b in range(3) for c in range(3)]
    checks = {}

    # rho_1: support and the 2x3 block
    X1 = [ix(l) for l in labels if l[1:] in ("00", "11", "22")]
    off_outside = [(i, j) for (i, j) in rho1.e if i != j and not (i in X1 and j in X1)]
    checks["rho1: off-diagonal entries only in C2 x span{00,11,22}"] = not off_outside
    checks["rho1: 2x3 block PSD"] = ex_is_psd(_sub(rho1, X1))
    checks["rho1: 2x3 block PPT"] = ex_is_psd(_sub(ex_partial_transpose(rho1, dims, (0,)), X1))
    checks["rho1: remaining diagonal >= 0"] = all(rho1.e.get((k, k), QSqrt(0, 0, 2)).sign() >= 0
                                                  for k in range(18) if k not in X1)

    # rho_2: block structure
    S2 = [ix(l) for l in labels if l[:2] in ("01", "02")]
    S1 = [ix(l) for l in labels if l[:2] in ("00", "11", "12")]
    cross = [(i, j) for (i, j) in rho2.e if (i in S1) != (j in S1)]
    outside = [(i, j) for (i, j) in rho2.e if i not in S1 + S2 or j not in S1 + S2]
    checks["rho2: no coherences between the two blocks"] = not cross and not outside
    checks["rho2: 2x3 block PSD"] = ex_is_psd(_sub(rho2, S2))
    checks["rho2: 2x3 block PPT"] = ex_is_psd(_sub(ex_partial_transpose(rho2, dims, (2,)), S2))
    corr = ["000", "111", "122"]; xs = ["00", "11", "12"]
    corr_idx = {ix(c) for c in corr}
    extra = [(i, j) for (i, j) in rho2.e if i != j and i in S1 and j in S1
             and not (i in corr_idx and j in corr_idx)]
    checks["rho2: maximally correlated form"] = not extra
    A = _sub(rho2, [ix(c) for c in corr])
    checks["rho2: A PSD"] = ex_is_psd(A)
    zero = QSqrt(0, 0, 2)
    checks["rho2: A entrywise >= 0"] = all(A.e.get((i, j), zero).sign() >= 0 for i in range(3) for j in range(3))
    ok_ab = True
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            a = A.e.get((i, j), zero)
            b_ij = rho2.e.get((ix(xs[i] + str(j)), ix(xs[i] + str(j))), zero)
            b_ji = rho2.e.get((ix(xs[j] + str(i)), ix(xs[j] + str(i))), zero)
            ok_ab &= (b_ij - a).sign() >= 0 and (b_ji - a).sign() >= 0
    checks["rho2: A_ij <= B_ij, B_ji"] = ok_ab
    checks["decomposition valid (sum, PSD, PT-PSD)"] = prim["all_ok"]
    return dict(checks=checks, biseparable=all(checks.values()), p_star=prim["p_star"])
