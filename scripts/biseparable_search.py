"""NUMERICAL search for biseparable decompositions of rho_d(p) by column generation
(supporting the d = 4 bracket in the paper; NOT an exact certificate -- the exact d = 3 proof is
gme332.certificates.biseparability_certificate_d3).

Reference output: results/reference/biseparable_search_d4.log  (python scripts/biseparable_search.py 4 45, ~22 min).

Certified biseparable decompositions by column generation.

Inner set = cone of (i) PPT pieces on 2x3 / 3x2 coordinate product subspaces (separable by
Horodecki) and (ii) atoms twirl(|a><a| (x) |b><b|) across a cut (separable: mixtures of
product states; twirling over the local diagonal unitaries that fix psi_d keeps them
separable and maps rho_d(p) to itself).  Atoms are added by alternating maximisation of the
dual-weighted overlap.  Any feasible p is a proven upper bound on the GME threshold
(up to the numerical tolerance of the final decomposition, checked separately).

    python colgen_biseparable.py [d] [iterations]
"""
import itertools, sys, time, pickle
import numpy as np
import cvxpy as cp

d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rng = np.random.default_rng(0)
dims = [2, d, d]; D = 2 * d * d
basis = list(itertools.product(range(2), range(d), range(d)))
idx = {s: i for i, s in enumerate(basis)}
psi = np.zeros(D); psi[idx[(0, 0, 0)]] = 1
for j in range(1, d): psi[idx[(1, j, j)]] = 1
psi /= np.linalg.norm(psi); rho = np.outer(psi, psi)

# torus charges: random phases satisfying the stabiliser constraints
ph = []
for _ in range(3):
    a0, a1, th = rng.normal(size=3); be = rng.normal(size=d)
    ga = [th - a0 - be[0]] + [th - a1 - be[j] for j in range(1, d)]
    ph.append(np.array([[a0, a1][a] + be[b] + ga[c] for a, b, c in basis]))
SAME = np.array([[all(abs(q[i] - q[j]) < 1e-9 for q in ph) for j in range(D)] for i in range(D)])
twirl = lambda M: M * SAME

CUTS = {"A|BC": ((0,), (1, 2)), "AB|C": ((0, 1), (2,)), "AC|B": ((0, 2), (1,))}
def side(parties): return list(itertools.product(*[range(dims[p]) for p in parties]))
def gidx(X, x, Y, y):
    s = [0, 0, 0]
    for p, v in zip(X, x): s[p] = v
    for p, v in zip(Y, y): s[p] = v
    return idx[tuple(s)]
EMB = {}
for name, (X, Y) in CUTS.items():
    SX, SY = side(X), side(Y)
    EMB[name] = (len(SX), len(SY), np.array([[gidx(X, x, Y, y) for y in SY] for x in SX]))

def atom(name, a, b):
    nx, ny, G = EMB[name]
    v = np.zeros(D)
    for i in range(nx):
        for j in range(ny): v[G[i, j]] = a[i] * b[j]
    return twirl(np.outer(v, v))

# coordinate PPT pieces (as in inner_biseparable_sdp.py)
pieces = []
for name, (X, Y) in CUTS.items():
    SX, SY = side(X), side(Y)
    for nx, ny in ((2, 3), (3, 2)):
        if nx > len(SX) or ny > len(SY): continue
        for U in itertools.combinations(SX, nx):
            for V in itertools.combinations(SY, ny):
                g, loc = [], []
                for i, u in enumerate(U):
                    for j, v in enumerate(V):
                        g.append(gidx(X, u, Y, v)); loc.append((i, j))
                if not any(SAME[a, b] for a, b in itertools.combinations(g, 2)):
                    continue                       # cannot carry coherences: diagonal atoms cover it
                pieces.append((g, loc))
def pt_perm(loc):
    pos = {l: k for k, l in enumerate(loc)}; n = len(loc); perm = np.zeros(n * n, dtype=int)
    for a, (i, j) in enumerate(loc):
        for b, (i2, j2) in enumerate(loc):
            perm[pos[(i, j2)] * n + pos[(i2, j)]] = a * n + b
    return perm

atoms = []      # (name, a, b, matrix)
def best_atoms(Y, n_restarts=30):
    """Maximise and minimise <Y, twirl(ab ab^T)> over unit a, b for each cut (alternating eigvec)."""
    Yt = twirl(Y); found = []
    for name in CUTS:
        nx, ny, G = EMB[name]
        for sign in (+1, -1):
            best = None
            for _ in range(n_restarts):
                b = np.abs(rng.normal(size=ny)); b /= np.linalg.norm(b)
                for _ in range(60):
                    # fix b: quadratic form in a; fix a: in b  (Yt is twirled so twirl(vv^T) overlap = v^T Yt v)
                    Ma = np.array([[sum(Yt[G[i, j], G[k, l]] * b[j] * b[l] for j in range(ny) for l in range(ny))
                                    for k in range(nx)] for i in range(nx)])
                    w, V = np.linalg.eigh(sign * Ma); a = V[:, -1]
                    Mb = np.array([[sum(Yt[G[i, j], G[k, l]] * a[i] * a[k] for i in range(nx) for k in range(nx))
                                    for l in range(ny)] for j in range(ny)])
                    w, V = np.linalg.eigh(sign * Mb); b = V[:, -1]
                val = sign * float(np.vdot(np.kron(1, 1), 1)) * 0 + sign * (atom(name, a, b) * Yt).sum()
                if best is None or val > best[0]: best = (val, a.copy(), b.copy())
            found.append((name, best[1], best[2], best[0]))
    return found

def solve():
    p = cp.Variable(); total = 0; cons = []
    for g, loc in pieces:
        n = len(g); T = cp.Variable((n, n), symmetric=True)
        cons += [T >> 0, cp.reshape(cp.vec(T, order="C")[pt_perm(loc)], (n, n), order="C") >> 0]
        E = np.zeros((D, n)); E[g, np.arange(n)] = 1; total = total + E @ T @ E.T
    dg = cp.Variable(D, nonneg=True); total = total + cp.diag(dg)        # diagonal product states
    w = cp.Variable(len(atoms), nonneg=True) if atoms else None
    if atoms:
        total = total + sum(w[k] * atoms[k][3] for k in range(len(atoms)))
    eq = total == (1 - p) * rho + p * np.eye(D) / D
    prob = cp.Problem(cp.Minimize(p), cons + [eq]); prob.solve(solver="CLARABEL")
    return prob.status, float(p.value), eq.dual_value, (w.value if atoms else None)

pstar = 2 * d * (d - 1) / (2 * (d * d - 1) + np.sqrt(d - 1))
if __name__ == "__main__":
  t0 = time.time()
  for it in range(ITERS):
      st, pv, Y, wv = solve()
      print(f"iter {it:2d}: {len(atoms):4d} atoms  status {st}  p = {pv:.8f}  gap to PPT bound {pv - pstar:+.2e}  ({time.time()-t0:.0f}s)", flush=True)
      if pv - pstar < 1e-6: break
      for name, a, b, val in best_atoms(np.asarray(Y)):
          atoms.append((name, a, b, atom(name, a, b)))
  pickle.dump(dict(atoms=[(n, a, b) for n, a, b, _ in atoms], p=pv, weights=wv),
              open(f"biseparable_search_d{d}.pkl", "wb"))
