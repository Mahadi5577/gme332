"""Exact arithmetic in Q(sqrt m) and exact positive-semidefiniteness tests.

Pure Python (fractions); NumPy is imported only by ExMat.to_numpy()."""
import itertools
import math
from fractions import Fraction
from typing import Dict

from .basis import flat_index


class QSqrt:
    """Exact number a + b*sqrt(m) with rational a, b (m a fixed positive integer)."""
    __slots__ = ("a", "b", "m")

    def __init__(self, a=0, b=0, m=2):
        self.a, self.b, self.m = Fraction(a), Fraction(b), int(m)

    def _c(self, o):
        return o if isinstance(o, QSqrt) else QSqrt(o, 0, self.m)

    def __add__(self, o):
        o = self._c(o); return QSqrt(self.a + o.a, self.b + o.b, self.m)
    __radd__ = __add__

    def __neg__(self):
        return QSqrt(-self.a, -self.b, self.m)

    def __sub__(self, o):
        return self + (-self._c(o))

    def __rsub__(self, o):
        return self._c(o) - self

    def __mul__(self, o):
        o = self._c(o)
        return QSqrt(self.a * o.a + self.m * self.b * o.b,
                     self.a * o.b + self.b * o.a, self.m)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._c(o)
        den = o.a * o.a - o.m * o.b * o.b
        return self * QSqrt(o.a / den, -o.b / den, self.m)

    def sign(self) -> int:
        a, b, m = self.a, self.b, self.m
        sa, sb = (a > 0) - (a < 0), (b > 0) - (b < 0)
        if sb == 0: return sa
        if sa == 0 or sa == sb: return sb if sa == 0 else sa
        return sa if a * a > m * b * b else sb

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(self.m)

    def __repr__(self):
        return f"{self.a} + {self.b}*sqrt({self.m})"


class ExMat:
    """Sparse exact symmetric matrix {(i, j): QSqrt} of size n (real entries)."""

    def __init__(self, n: int, m: int):
        self.n, self.m, self.e = n, m, {}

    def add(self, i, j, v):
        v = v if isinstance(v, QSqrt) else QSqrt(v, 0, self.m)
        self.e[(i, j)] = self.e.get((i, j), QSqrt(0, 0, self.m)) + v
        if self.e[(i, j)].is_zero():
            del self.e[(i, j)]
        return self

    def add_sym(self, i, j, v):
        self.add(i, j, v)
        return self if i == j else self.add(j, i, v)

    def add_outer(self, vec: Dict[int, object], w=1):
        for i, vi in vec.items():
            for j, vj in vec.items():
                self.add(i, j, QSqrt(w, 0, self.m) * vi * vj if not isinstance(w, QSqrt)
                         else w * vi * vj)
        return self

    def copy(self):
        out = ExMat(self.n, self.m); out.e = dict(self.e); return out

    def __add__(self, o):
        out = self.copy()
        for k, v in o.e.items(): out.add(*k, v)
        return out

    def __sub__(self, o):
        out = self.copy()
        for k, v in o.e.items(): out.add(*k, -v)
        return out

    def scale(self, c):
        out = ExMat(self.n, self.m)
        for k, v in self.e.items(): out.add(*k, v * c)
        return out

    def trace(self):
        return sum((v for (i, j), v in self.e.items() if i == j), QSqrt(0, 0, self.m))

    def inner(self, o):
        """Tr(self . o) for symmetric matrices."""
        return sum((v * o.e[(j, i)] for (i, j), v in self.e.items() if (j, i) in o.e),
                   QSqrt(0, 0, self.m))

    def equals(self, o) -> bool:
        return not (self - o).e

    def to_numpy(self):
        import numpy as np
        M = np.zeros((self.n, self.n))
        for (i, j), v in self.e.items(): M[i, j] = float(v)
        return M


def ex_partial_transpose(M: ExMat, dims, parties) -> ExMat:
    dims = list(dims)
    def unflat(k):
        out = []
        for dk in reversed(dims):
            out.append(k % dk); k //= dk
        return out[::-1]
    out = ExMat(M.n, M.m)
    for (i, j), v in M.e.items():
        r, c = unflat(i), unflat(j)
        for p in parties: r[p], c[p] = c[p], r[p]
        out.add(flat_index(r, dims), flat_index(c, dims), v)
    return out


def ex_swap_BC(M: ExMat, d: int) -> ExMat:
    dims = [2, d, d]
    def sw(k):
        a, rem = divmod(k, d * d); b, c = divmod(rem, d)
        return flat_index((a, c, b), dims)
    out = ExMat(M.n, M.m)
    for (i, j), v in M.e.items(): out.add(sw(i), sw(j), v)
    return out


def _ex_det(B):
    n = len(B)
    if n == 1: return B[0][0]
    tot = QSqrt(0, 0, B[0][0].m)
    for c in range(n):
        if B[0][c].is_zero(): continue
        minor = [row[:c] + row[c + 1:] for row in B[1:]]
        term = B[0][c] * _ex_det(minor)
        tot = tot + term if c % 2 == 0 else tot - term
    return tot


def ex_is_psd(M: ExMat) -> bool:
    """Exact PSD test: on every connected block, all elementary symmetric
    functions of the eigenvalues (sums of k x k principal minors) are >= 0."""
    adj = {i: set() for i in range(M.n)}
    for (i, j) in M.e: adj[i].add(j); adj[j].add(i)
    seen = set()
    for s in range(M.n):
        if s in seen: continue
        comp, stack = [], [s]
        while stack:
            k = stack.pop()
            if k in seen: continue
            seen.add(k); comp.append(k); stack += list(adj[k] - seen)
        comp.sort()
        zero = QSqrt(0, 0, M.m)
        B = [[M.e.get((i, j), zero) for j in comp] for i in comp]
        for k in range(1, len(comp) + 1):
            ek = zero
            for sub in itertools.combinations(range(len(comp)), k):
                ek = ek + _ex_det([[B[i][j] for j in sub] for i in sub])
            if ek.sign() < 0:
                return False
    return True
