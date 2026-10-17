"""Construct complete digit systems directly from a Z_(p^r) check.

The paper treats Galois rings; this independent implementation verifies its
integer-residue-ring specialization. Metric tests remain in original coordinates.
"""


def identity(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def matvec(a, x, modulus):
    return [sum(u*v for u, v in zip(row, x)) % modulus for row in a]


def smith_check(h, p, r):
    modulus, rows, cols = p**r, len(h), len(h[0])
    a = [[x % modulus for x in row] for row in h]
    u, v, inverse_v = identity(rows), identity(cols), identity(cols)

    def valuation(x):
        result = 0
        while result < r and x % p == 0:
            x //= p
            result += 1
        return result

    values = []
    for i in range(min(rows, cols)):
        value, pi, pj = min((valuation(a[j][k]), j, k)
                           for j in range(i, rows) for k in range(i, cols))
        if value == r:
            break
        a[i], a[pi] = a[pi], a[i]
        u[i], u[pi] = u[pi], u[i]
        for matrix in (a, v):
            for row in matrix:
                row[i], row[pj] = row[pj], row[i]
        inverse_v[i], inverse_v[pj] = inverse_v[pj], inverse_v[i]
        scale = pow(a[i][i]//p**value, -1, modulus)
        a[i] = [scale*x % modulus for x in a[i]]
        u[i] = [scale*x % modulus for x in u[i]]
        for j in range(i+1, rows):
            factor = a[j][i]//p**value
            a[j] = [(x-factor*y) % modulus for x, y in zip(a[j], a[i])]
            u[j] = [(x-factor*y) % modulus for x, y in zip(u[j], u[i])]
        for j in range(i+1, cols):
            factor = a[i][j]//p**value
            for matrix in (a, v):
                for row in matrix:
                    row[j] = (row[j]-factor*row[i]) % modulus
            inverse_v[i] = [(x+factor*y) % modulus for x, y in zip(inverse_v[i], inverse_v[j])]
        values.append(value)
    return dict(p=p, r=r, h=h, u=u, v=v, inverse_v=inverse_v, values=values, diagonal=a)


def digit_system(prepared, syndrome, prefix, level):
    p, r = prepared['p'], prepared['r']
    if not 0 <= level < r:
        raise ValueError('Invalid digit level')
    modulus = p**r
    residual = [(a-b) % modulus for a, b in zip(syndrome, matvec(prepared['h'], prefix, modulus))]
    beta = matvec(prepared['u'], residual, modulus)
    h, target = [], []
    for j, value in enumerate(prepared['values']):
        precision = level+value
        if beta[j] % p**min(r, precision):
            raise ValueError('Prefix has no linear completion')
        if precision < r:
            h.append([x % p for x in prepared['inverse_v'][j]])
            target.append(beta[j]//p**precision % p)
    if any(beta[len(prepared['values']):]):
        raise ValueError('Inconsistent zero row')
    return h, target
