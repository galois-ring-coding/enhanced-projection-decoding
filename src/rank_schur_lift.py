"""Rank-stable Schur lifting over GR(2**r,m); no rank-SD oracle is hidden.

Large cases benchmark ONLY lifting, with a supplied correct residue error.
The small repetition-like code also decodes that residue by exhaustive search.
"""
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np


def poly_mod(a, f):
    while a.bit_length() >= f.bit_length():
        a ^= f << (a.bit_length() - f.bit_length())
    return a


def gf_mul(a, b, f):
    out = 0
    degree = f.bit_length() - 1
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a >> degree:
            a ^= f
    return out


def irreducible(f):
    m = f.bit_length() - 1
    x = 2
    for j in range(1, m + 1):
        x = gf_mul(x, x, f)
        if j <= m // 2:
            a, b = f, x ^ 2
            while b:
                a, b = b, poly_mod(a, b)
            if a != 1:
                return False
    return x == 2


def polynomial(m):
    known = {3: 0b1011, 32: (1 << 32) | 0x8d,
             48: (1 << 48) | 0x2d, 64: (1 << 64) | 0x1b}
    f = known[m]
    assert irreducible(f), (m, hex(f))
    return f


def bits(v):
    return sum((int(x) & 1) << j for j, x in enumerate(v))


def unbits(x, m):
    return [(x >> j) & 1 for j in range(m)]


def ring_mul(a, b, modulus, f):
    m = len(a)
    width = (m * (modulus - 1)**2).bit_length()
    aa = sum((int(x) % modulus) << (width*j) for j, x in enumerate(a))
    bb = sum((int(x) % modulus) << (width*j) for j, x in enumerate(b))
    product = aa * bb
    mask = (1 << width) - 1
    out = [(product >> (width*j)) & mask for j in range(2*m-1)]
    terms = [j for j in range(m) if (f >> j) & 1]
    for j in range(2*m-2, m-1, -1):
        for aindex in terms:
            out[j-m+aindex] -= out[j]
    return np.array([x % modulus for x in out[:m]], dtype=np.int64)


def syndrome(h, e, modulus, f):
    out = np.zeros((len(e), len(h)), dtype=np.int64)
    for a, row in enumerate(h):
        for j, hj in enumerate(row):
            out[:, a] += ring_mul(hj, e[:, j], modulus, f)
        out[:, a] %= modulus
    return out


def inverse(a, modulus):
    n = len(a)
    b = np.concatenate((a.copy() % modulus, np.eye(n, dtype=np.int64)), axis=1)
    for j in range(n):
        k = next(k for k in range(j, n) if b[k, j] % 2)
        b[[j, k]] = b[[k, j]]
        b[j] = b[j] * pow(int(b[j, j]), -1, modulus) % modulus
        for k in range(n):
            if k != j:
                b[k] = (b[k] - b[k, j] * b[j]) % modulus
    return b[:, n:]


def pivot_order(e, t):
    a = e.copy() % 2
    m, n = a.shape
    rows, cols = list(range(m)), list(range(n))
    for j in range(t):
        pivot = next(((k, l) for k in range(j, m) for l in range(j, n) if a[k, l]), None)
        if pivot is None:
            raise ValueError("Residue rank is smaller than t; rank-stable lifting does not apply")
        k, l = pivot
        a[[j, k]] = a[[k, j]]
        a[:, [j, l]] = a[:, [l, j]]
        rows[j], rows[k] = rows[k], rows[j]
        cols[j], cols[l] = cols[l], cols[j]
        for k in range(j+1, m):
            if a[k, j]:
                a[k] ^= a[j]
    if a[t:, t:].any():
        raise ValueError("Residue rank exceeds t")
    return rows, cols


def pack_syndrome(hbar, vec, f, m):
    packed = 0
    for row in reversed(hbar):
        y = 0
        for a, b in zip(row, vec):
            y ^= gf_mul(a, b, f)
        packed = (packed << m) | y
    return packed


class ColumnSolver:
    """Cached GF(2) column elimination, retaining preimages for RHS solves."""
    def __init__(self, columns):
        self.basis = {}
        self.xors = 0
        self.nvars = len(columns)
        for j, value in enumerate(columns):
            mask = 1 << j
            while value:
                p = value.bit_length() - 1
                if p not in self.basis:
                    self.basis[p] = (value, mask)
                    break
                old, preimage = self.basis[p]
                value ^= old
                mask ^= preimage
                self.xors += 2
        if len(self.basis) != self.nvars:
            raise ValueError("Nonunique tangent system; branching is required")

    def solve(self, target):
        mask = 0
        while target:
            p = target.bit_length() - 1
            if p not in self.basis:
                raise ValueError("Inconsistent tangent system")
            row, preimage = self.basis[p]
            target ^= row
            mask ^= preimage
            self.xors += 2
        return mask


def lift(h, s, residue, t, r, f):
    start = time.perf_counter()
    m, n = residue.shape
    if r < 1 or not 0 <= t <= min(m, n):
        raise ValueError('Invalid precision or target rank')
    if np.any((residue < 0) | (residue > 1)):
        raise ValueError('Residue coefficients must be binary')
    # The mathematical theorem has no such restriction; this NumPy kernel does.
    if max(m, n, 1)**2 * (2**r-1)**3 >= np.iinfo(np.int64).max:
        raise ValueError('Parameters exceed the certified int64 arithmetic range')
    rows, cols = pivot_order(residue, t)
    hp = [[row[j] for j in cols] for row in h]
    hbar = [[bits(x) for x in row] for row in hp]
    e = residue[np.ix_(rows, cols)].copy()
    a_inv = inverse(e[:t, :t], 2)
    b0 = (a_inv @ e[:t, t:]) % 2
    c0 = (e[t:, :t] @ a_inv) % 2
    low_basis = [1 << rows[j] for j in range(m)]
    ccols = [sum(int(c0[j, a]) << rows[t+j] for j in range(m-t)) for a in range(t)]
    w = [[0]*t for _ in hp]
    for j, row in enumerate(hbar):
        for a in range(t):
            for c in range(n-t):
                if b0[a, c]:
                    w[j][a] ^= row[t+c]
    columns = []
    for a in range(t):
        for b in range(t):
            columns.append(sum((gf_mul(row[b], low_basis[a], f) ^
                                gf_mul(w[j][b], ccols[a], f)) << (m*j)
                               for j, row in enumerate(hbar)))
    for a in range(t):
        for c in range(n-t):
            columns.append(sum(gf_mul(row[t+c], low_basis[a] ^ ccols[a], f) << (m*j)
                               for j, row in enumerate(hbar)))
    for b in range(m-t):
        for a in range(t):
            columns.append(sum(gf_mul(row[a] ^ w[j][a], low_basis[t+b], f) << (m*j)
                               for j, row in enumerate(hbar)))
    solver = ColumnSolver(columns)
    factor_time = time.perf_counter() - start
    for i in range(1, r):
        mod, scale = 2**(i+1), 2**i
        schur = (e[t:, t:] - e[t:, :t] @ inverse(e[:t, :t], mod) @ e[:t, t:]) % mod
        if (schur % scale).any():
            raise ValueError('Prefix is not rank-stable at the claimed precision')
        residual = (schur // scale) % 2
        fixed = [0]*t + [sum(int(residual[b, c]) << rows[t+b]
                                    for b in range(m-t)) for c in range(n-t)]
        original_rows = e[np.argsort(rows), :]
        diff = (s - syndrome(hp, original_rows, mod, f)) % mod
        if (diff % scale).any():
            raise ValueError('Prefix does not satisfy the lower-precision syndrome')
        target = sum(bits(diff[:, j] // scale) << (m*j) for j in range(len(h)))
        target ^= pack_syndrome(hbar, fixed, f, m)
        solution = solver.solve(target)
        v = np.array(unbits(solution, solver.nvars), dtype=np.int64)
        x = v[:t*t].reshape(t, t)
        y = v[t*t:t*n].reshape(t, n-t)
        z = v[t*n:].reshape(m-t, t)
        bottom = (z @ b0 + c0 @ y - c0 @ x @ b0 - residual) % 2
        digit = np.block([[x, y], [z, bottom]])
        e += scale * digit
        if ((e[t:, t:] - e[t:, :t] @ inverse(e[:t, :t], mod) @ e[:t, t:]) % mod).any():
            raise ValueError('Recovered prefix violates the rank constraint')
    result = e[np.ix_(np.argsort(rows), np.argsort(cols))]
    if not np.array_equal(syndrome(h, result, 2**r, f), s):
        raise ValueError('Final original syndrome verification failed')
    return result, {"factor_seconds": factor_time, "lift_seconds": time.perf_counter()-start,
                    "unknowns": solver.nvars, "equations": len(h)*m,
                    "tangent_rank": len(solver.basis), "packed_xors": solver.xors,
                    "exponential_candidates": 0}


def random_case(m, n, k, t, seed, r=4):
    rng = np.random.default_rng(seed)
    f, modulus = polynomial(m), 2**r
    u = rng.integers(0, modulus, (m, t), dtype=np.int64)
    v = rng.integers(0, modulus, (t, n), dtype=np.int64)
    u[:t] = np.eye(t, dtype=np.int64)
    v[:, :t] = np.eye(t, dtype=np.int64)
    e = (u @ v) % modulus
    e = e[np.ix_(rng.permutation(m), rng.permutation(n))]
    h = rng.integers(0, modulus, (n-k, n, m), dtype=np.int64)
    h[:, k:, :] = 0
    for j in range(n-k):
        h[j, k+j, 0] = 1
    start = time.perf_counter()
    s = syndrome(h, e, modulus, f)
    generation_seconds = time.perf_counter()-start
    recovered, stats = lift(h, s, e % 2, t, r, f)
    assert np.array_equal(recovered, e)
    return dict(m=m, n=n, k=k, t=t, r=r, seed=seed, polynomial=hex(f),
                correct=True, initial_residue_supplied=True,
                distance_verified=False, syndrome_generation_seconds=generation_seconds, **stats)


def small_end_to_end(seed, valuation=0):
    rng = random.Random(seed)
    f, m, n, t, r = polynomial(3), 3, 3, 1, 4
    modulus = 2**r
    if not 0 <= valuation < r:
        raise ValueError('Code generator valuation must be in [0,r)')
    h = np.zeros((2+bool(valuation), 3, 3), dtype=np.int64)
    h[0, 0, 1] = modulus-1
    h[0, 1, 0] = 1
    h[1, 0, 2] = modulus-1
    h[1, 2, 0] = 1
    if valuation:
        h[2, 0, 0] = 2**(r-valuation)
    unit = np.array([rng.randrange(modulus) for _ in range(m)], dtype=np.int64)
    unit[0] |= 1
    coefficients = np.array([rng.randrange(modulus) for _ in range(n)], dtype=np.int64)
    coefficients[1] |= 1
    e = unit[:, None] * coefficients[None, :] % modulus
    # A received representative is obtained from the systematic check, not e.
    s = syndrome(h, e, modulus, f)
    first = s[:, 2]//2**(r-valuation) if valuation else np.zeros(m, dtype=np.int64)
    received = np.column_stack((first,
        (s[:, 0]+ring_mul(first, [0,1,0], modulus, f)) % modulus,
        (s[:, 1]+ring_mul(first, [0,0,1], modulus, f)) % modulus))
    if not np.array_equal(syndrome(h, received, modulus, f), s):
        raise ValueError('Public representative violates the check')
    received %= 2
    candidates = []
    for a in range(1 if valuation else 8):
        word = np.array([unbits(gf_mul(a, b, f), m) for b in (1, 2, 4)]).T
        prefix = (received - word) % 2
        from verify_theory import rank_chain
        if rank_chain(prefix.tolist(), 2, 1) <= 1:
            candidates.append(prefix)
    assert len(candidates) == 1
    recovered, stats = lift(h, s, candidates[0], t, r, f)
    assert np.array_equal(recovered, e)
    return dict(seed=seed, correct=True, generator_valuation=valuation,
                residue_codewords_enumerated=1 if valuation else 8,
                initial_residue_supplied=False, distance=3, **stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    small = [small_end_to_end(24092026+j) for j in range(20)]
    nonfree = [small_end_to_end(24092026+j, valuation=1) for j in range(20)]
    large = []
    if not args.small_only:
        for m, n, k, t in ((32, 64, 24, 4), (48, 96, 32, 6), (64, 128, 48, 8)):
            for j in range(3):
                row = random_case(m, n, k, t, 24092026+j)
                large.append(row)
                print(json.dumps(row), flush=True)
    data = dict(python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), small=small, small_nonfree=nonfree, large=large,
                scope="Large results measure rank-stable lifting, NOT complete rank SD")
    destination = args.output or Path(__file__).with_name(
        'rank_schur_small_results.json' if args.small_only else 'rank_schur_results.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('small:', len(small), 'nonfree:', len(nonfree), 'large:', len(large))


if __name__ == '__main__':
    main()
