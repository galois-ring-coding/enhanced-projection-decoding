"""Paired complete Z4 Lee-SD comparison, with certified projected uniqueness.

DCC6 keeps Algorithm 6's recovery rule; both first-stage solvers use the
public bound t instead of an instance-unknown expected projected weight.
SC uses the exact identity wt_L(d0+2*d1)=|supp(d0)|+2*wt(d1 outside supp(d0)).
Only code-dependent matrices are cached. All kernels use Prange (v=0).
"""
import argparse
import csv
import json
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from ring_comparison_suite import Code


METHODS = ('dcc6', 'super_support')
TRIAL_LIMIT = 10000


def lee(x):
    return sum((0, 1, 2, 1)[int(a) % 4] for a in x)


def raw_syndrome(h, x):
    return [sum(a*b for a, b in zip(row, x)) % 4 for row in h]


class Arithmetic:
    """Dense scalar operations; counts are distinct from uninstrumented timings."""
    def __init__(self, bits, track=True):
        self.bits, self.mod, self.track = bits, 2**bits, track
        self.stats = Counter()

    def event(self, name, count=1):
        if self.track:
            self.stats[name] += count

    def add(self, x, y):
        if self.track:
            self.stats['adds'] += 1
            self.stats['arithmetic_proxy'] += self.bits
        return (x+y) % self.mod

    def mul(self, x, y):
        if self.track:
            self.stats['mults'] += 1
            self.stats['arithmetic_proxy'] += self.bits**2
        return x*y % self.mod

    def inv(self, x):
        if self.track:
            self.stats['inversions'] += 1
            self.stats['arithmetic_proxy'] += self.bits**3
        return pow(x, -1, self.mod)

    def dot(self, row, vector):
        if self.track:
            count = len(row)
            self.stats['adds'] += count
            self.stats['mults'] += count
            self.stats['arithmetic_proxy'] += count*(self.bits+self.bits**2)
        return sum(a*b for a, b in zip(row, vector)) % self.mod


def eliminate(h, target, selected, op):
    """Apply unit pivots to the whole augmented check, including torsion rows."""
    a = [[x % op.mod for x in row]+[b % op.mod] for row, b in zip(h, target)]
    for r, col in enumerate(selected):
        pivot = next((j for j in range(r, len(a)) if a[j][col] % 2), None)
        if pivot is None:
            return None
        a[r], a[pivot] = a[pivot], a[r]
        factor = op.inv(a[r][col])
        a[r] = [op.mul(x, factor) for x in a[r]]
        for j in range(len(a)):
            if j != r:
                factor = a[j][col]
                a[j] = [op.add(x, -op.mul(factor, y)) for x, y in zip(a[j], a[r])]
    return a


def prange(h, target, radius, rng, op):
    n, size = len(h[0]), len(h)
    if not any(target):
        op.event('zero_syndrome_shortcuts')
        return [0]*n
    if size == n and all(row == [int(j == i) for j in range(n)] for i, row in enumerate(h)):
        op.event('identity_shortcuts')
        op.event('candidates')
        if sum(target) <= radius:
            return list(target)
        raise ValueError('No admissible identity-system solution')
    for _ in range(TRIAL_LIMIT):
        op.event('trials')
        selected = rng.sample(range(n), size)
        a = eliminate(h, target, selected, op)
        if a is None:
            op.event('singular_trials')
            continue
        op.event('candidates')
        e = [0]*n
        for j, row in zip(selected, a):
            e[j] = row[-1]
        if sum(e) <= radius:
            return e
    raise RuntimeError('Prange trial limit reached')


def support_digit(h, target, known, radius, rng, op):
    """Eliminate known locations and solve exactly the remaining binary weight problem."""
    n, s = len(h[0]), len(known)
    rest = [j for j in range(n) if j not in known]
    a = eliminate(h, target, known, op)
    if a is None:
        raise ValueError('Dependent known support under the claimed distance certificate')
    reduced = [[row[j] for j in rest] for row in a[s:]]
    beta = [row[-1] for row in a[s:]]
    if radius == 0:
        op.event('zero_budget_shortcuts')
        if any(beta):
            raise ValueError('Known support cannot complete the Lee budget')
        unknown = [0]*len(rest)
    else:
        unknown = prange(reduced, beta, radius, rng, op)
    digit = [0]*n
    for j, x in zip(rest, unknown):
        digit[j] = x
    for r, j in enumerate(known):
        digit[j] = op.add(a[r][-1], -op.dot([a[r][k] for k in rest], unknown))
    return digit


def prepare(code, method):
    h = code.parity(2)
    prepared = dict(code=code, h=h, method=method, unit_rows=code.n-len(code.g))
    if method == 'dcc6':
        prepared['h_first'] = [[x % 2 for x in row] for row in h[:prepared['unit_rows']]]
    else:
        prepared['h_first'], prepared['active'] = code.field_check(0)
        prepared['h_second'], _ = code.field_check(1)
    prepared['matrix_payload_bits'] = 2*sum(len(row) for row in h)
    prepared['matrix_payload_bits'] += sum(len(row) for row in prepared['h_first'])
    if method != 'dcc6':
        prepared['matrix_payload_bits'] += sum(len(row) for row in prepared['h_second'])
        prepared['matrix_payload_bits'] += 2*code.n*len(prepared['active'])
    return prepared


def received_representative(prepared, syndrome, op):
    code, hcount = prepared['code'], prepared['unit_rows']
    k, n = len(code.g), code.n
    y = [0]*n
    index = hcount
    for j, valuation in enumerate(code.valuations):
        if valuation:
            if syndrome[index] % 2:
                raise ValueError('Inconsistent torsion syndrome')
            y[j] = syndrome[index]//2
            index += 1
    for j in range(k, n):
        y[j] = op.add(syndrome[j-k], op.dot([row[j] for row in code.g], y[:k]))
    return y


def decode(prepared, syndrome, t, seed, track=True):
    code, method = prepared['code'], prepared['method']
    n, k = code.n, len(code.g)
    first, second, ring, verify = [Arithmetic(bits, track) for bits in (1, 1, 2, 2)]
    if method == 'dcc6':
        target = [x % 2 for x in syndrome[:prepared['unit_rows']]]
    else:
        y = received_representative(prepared, syndrome, ring)
        target = [first.dot(row, y) for row in prepared['h_first']]
    low = prange(prepared['h_first'], target, t, random.Random(seed*17), first)
    known = [j for j, x in enumerate(low) if x]
    if method == 'dcc6':
        rng = random.Random(seed*17+1)
        rest = [j for j in range(n) if j not in known]
        for _ in range(TRIAL_LIMIT):
            ring.event('trials')
            selected = known+rng.sample(rest, n-k-len(known))
            a = eliminate(prepared['h'], syndrome, selected, ring)
            if a is None:
                ring.event('singular_trials')
                continue
            if any(row[-1] for row in a[n-k:]):
                ring.event('inconsistent_trials')
                continue
            ring.event('candidates')
            answer = [0]*n
            for j, row in zip(selected, a):
                answer[j] = row[-1]
            if lee(answer) <= t:
                break
        else:
            raise RuntimeError('DCC recovery trial limit reached')
    else:
        # Compatible supercode lift, including all ring carry arithmetic.
        code_digit = [first.add(x, -e) for x, e in zip(y, low)]
        active = prepared['active']
        lifted_word = [ring.dot([code.g[j][col] for j in active],
                               [code_digit[j] for j in active]) for col in range(n)]
        numerator = [ring.add(ring.add(x, -e), -c) for x,e,c in zip(y,low,lifted_word)]
        if any(x % 2 for x in numerator):
            raise ValueError('Nondivisible supercode residual')
        residual = [x//2 for x in numerator]
        target = [second.dot(row, residual) for row in prepared['h_second']]
        high = support_digit(prepared['h_second'], target, known, (t-len(known))//2,
                             random.Random(seed*17+1), second)
        answer = [ring.add(a, ring.mul(2,b)) for a,b in zip(low, high)]
    if [verify.dot(row, answer) for row in prepared['h']] != syndrome or lee(answer) > t:
        raise ValueError('Final original-ring verification failed')
    counts = Counter()
    for label, arithmetic in (('first', first), ('second', second), ('ring', ring), ('verify', verify)):
        counts.update(arithmetic.stats)
        for key, value in arithmetic.stats.items():
            counts[label+'_'+key] = value
    counts['known_positions'] = len(known)
    counts['remaining_twos'] = (t-len(known))//2
    return answer, dict(counts)


def binary_distance(g):
    words = [0]
    for row in g:
        packed = sum(x << j for j,x in enumerate(row))
        words += [x ^ packed for x in words]
    return min(x.bit_count() for x in words[1:])


def code_suite(mothers=4):
    out = []
    for n,k,t in ((10,3,2),(16,4,3),(24,6,4)):
        for index in range(mothers):
            seed = 20260924+1000*n+index
            rng = random.Random(seed)
            for attempts in range(100000):
                g = [[int(i == j) for j in range(k)]+[rng.randrange(2) for _ in range(n-k)]
                     for i in range(k)]
                dh = binary_distance(g)
                if dh >= 2*t+1:
                    break
            else:
                raise RuntimeError('Distance rejection sampler exhausted')
            for row in g:
                for j in range(k,n):
                    row[j] += 2*rng.randrange(2)
            for label,k0 in (('free',k),('mild',k-1),('mixed',k//2),('even',0)):
                code = Code(f'n{n}_mother{index}_{label}',g,(0,)*k0+(1,)*(k-k0),r=2)
                dl = min(lee(w) for w in code.words() if any(w))
                assert 2*t < dh and 2*t < dl
                out.append((code,dict(cluster=f'n{n}_mother{index}',seed=seed,
                            attempts=attempts+1,type=label,n=n,K=k,k0=k0,t=t,
                            d_binary=dh,d_lee=dl)))
    return out


def sphere_table(n,t):
    table = [[0]*(t+1) for _ in range(n+1)]
    table[0][0] = 1
    for length in range(1,n+1):
        for weight in range(t+1):
            table[length][weight] = sum(table[length-1][weight-w]
                                       for w in (0,1,2,1) if w <= weight)
    return table


def sample_sphere(n,t,rng,table):
    remaining, error = t, []
    for j in range(n):
        choice = rng.randrange(table[n-j][remaining])
        for x,w in enumerate((0,1,2,1)):
            amount = table[n-j-1][remaining-w] if w <= remaining else 0
            if choice < amount:
                error.append(x)
                remaining -= w
                break
            choice -= amount
    assert remaining == 0
    return error


def error_cases(n,t,count,seed,balanced=False):
    rng = random.Random(seed)
    if not balanced:
        table = sphere_table(n,t)
        return [(f'uniform_{j}',sample_sphere(n,t,rng,table)) for j in range(count)]
    cases = []
    for twos in range(t//2+1):
        odd = t-2*twos
        for j in range(count):
            error = [0]*n
            support = rng.sample(range(n),odd+twos)
            for col in support[:odd]:
                error[col] = rng.choice((1,3))
            for col in support[odd:]:
                error[col] = 2
            cases.append((f'odd{odd}_{j}',error))
    return cases


def summarize(records,code_records,bootstrap=2000):
    failed = [dict(code=r['code'], distribution=r['distribution'], case=r['case'],
                   method=r['method'], failure=r.get('failure', 'Incorrect recovery'))
              for r in records if not r['correct']]
    if failed:
        return dict(status='Incomplete decoders are not ranked', failures=failed, groups=[])
    summary = []
    for distribution in ('uniform','composition'):
        for kind in ('all','free','mild','mixed','even'):
            sample = [r for r in records if r['distribution']==distribution and
                      (kind == 'all' or r['type']==kind)]
            if not sample:
                continue
            row = dict(distribution=distribution,type=kind,cases=len(sample)//2)
            for method in METHODS:
                group = [r for r in sample if r['method']==method]
                row[method+'_failures'] = sum(not r['correct'] for r in group)
                for metric in ('arithmetic_proxy','candidates','trials','online_ns','amortized_ns',
                               'first_arithmetic_proxy','second_arithmetic_proxy','ring_arithmetic_proxy'):
                    row[method+'_'+metric] = statistics.mean(r.get(metric,0) for r in group)
            for metric in ('arithmetic_proxy','online_ns','amortized_ns'):
                row['dcc_over_super_'+metric] = row['dcc6_'+metric]/row['super_support_'+metric]
            summary.append(row)
    # Cluster resampling preserves all derived subtypes of each mother matrix.
    cluster_values = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row['distribution']=='uniform':
            cluster_values[row['cluster']][row['method']].append(row)
    clusters = list(cluster_values)
    rng = random.Random(24682468)
    intervals = {}
    for metric in ('arithmetic_proxy','online_ns'):
        means = {c: {method:statistics.mean(r[metric] for r in cluster_values[c][method])
                     for method in METHODS} for c in clusters}
        ratios = []
        for _ in range(bootstrap):
            chosen = [rng.choice(clusters) for _ in clusters]
            ratios.append(sum(means[c]['dcc6'] for c in chosen)/
                          sum(means[c]['super_support'] for c in chosen))
        ratios.sort()
        intervals[metric] = [ratios[int(.025*bootstrap)],ratios[int(.975*bootstrap)]]
    wins = {}
    for metric in ('arithmetic_proxy','online_ns'):
        number = 0
        for code in code_records:
            rows = [r for r in records if r['code']==code['code'] and r['distribution']=='uniform']
            means = {method:statistics.mean(r[metric] for r in rows if r['method']==method)
                     for method in METHODS}
            number += means['super_support'] < means['dcc6']
        wins[metric] = dict(super_wins=number,total_codes=len(code_records))
    return dict(groups=summary,cluster_bootstrap_intervals=intervals,per_code_wins=wins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mothers',type=int,default=4)
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--composition-samples',type=int,default=40)
    parser.add_argument('--timing-repeats',type=int,default=3)
    parser.add_argument('--output',default='lee_z4_results')
    args = parser.parse_args()
    if min(args.mothers, args.samples, args.composition_samples, args.timing_repeats) < 1:
        parser.error('All sample counts and timing repeats must be positive')
    destination = Path(__file__).parent/args.output
    destination.mkdir(parents=True, exist_ok=True)
    records,code_records = [],[]
    for code,description in code_suite(args.mothers):
        prepared,setup = {},{}
        for method in METHODS:
            times = []
            for _ in range(7):
                start = time.perf_counter_ns()
                prepared[method] = prepare(code,method)
                times.append(time.perf_counter_ns()-start)
            setup[method] = statistics.median(times)
        code_records.append(dict(code=code.name,g=code.g,valuations=code.valuations,
                                 setup_ns=setup,**description,
                                 payload_bits={m:prepared[m]['matrix_payload_bits'] for m in METHODS}))
        for distribution,count,balanced in (('uniform',args.samples,False),
                                             ('composition',args.composition_samples,True)):
            cases = error_cases(code.n,description['t'],count,description['seed']+199,balanced)
            for index,(case_id,error) in enumerate(cases):
                seed = description['seed']*100003+index+(10000000 if balanced else 0)
                syndrome = raw_syndrome(prepared['dcc6']['h'],error)
                current = {}
                order = METHODS if index % 2 == 0 else tuple(reversed(METHODS))
                for method in order:
                    row = dict(code=code.name,cluster=description['cluster'],type=description['type'],
                               n=code.n,k0=description['k0'],K=description['K'],t=description['t'],
                               distribution=distribution,case=case_id,seed=seed,method=method,
                               error=error,syndrome=syndrome,correct=False)
                    try:
                        answer,counts = decode(prepared[method],syndrome,description['t'],seed)
                        row.update(counts)
                        row['correct'] = answer == error
                        row['answer'] = answer
                    except (RuntimeError,ValueError) as exc:
                        row['failure'] = str(exc)
                    current[method] = row
                # Separate timings disable arithmetic counters, keeping identical trial streams.
                timed = defaultdict(list)
                for repeat in range(args.timing_repeats):
                    for method in (order if repeat % 2 == 0 else tuple(reversed(order))):
                        start = time.perf_counter_ns()
                        try:
                            answer,_ = decode(prepared[method],syndrome,description['t'],seed,track=False)
                            if answer != error:
                                current[method]['correct'] = False
                        except (RuntimeError,ValueError):
                            current[method]['correct'] = False
                        timed[method].append(time.perf_counter_ns()-start)
                for method in METHODS:
                    current[method]['online_ns'] = statistics.median(timed[method])
                    current[method]['setup_ns'] = setup[method]
                    current[method]['amortized_ns'] = current[method]['online_ns']+setup[method]/len(cases)
                    current[method]['cold_ns'] = current[method]['online_ns']+setup[method]
                    records.append(current[method])
        print(code.name,'verified',len(cases),'composition cases',flush=True)
    failures = sum(not row['correct'] for row in records)
    data = dict(config=vars(args),python=platform.python_version(),platform=platform.platform(),
                failures=failures,decoder_calls=len(records),timed_calls=len(records)*args.timing_repeats,
                code_count=len(code_records),codes=code_records,
                scope='Paired Z4 Lee SD; safe radius t in first stage; no support/answer cache',
                summary=summarize(records,code_records))
    (destination/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    with (destination/'instances.jsonl').open('w',encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row)+'\n')
    with (destination/'groups.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        if data['summary']['groups']:
            writer = csv.DictWriter(stream,fieldnames=list(data['summary']['groups'][0]))
            writer.writeheader()
            writer.writerows(data['summary']['groups'])
        else:
            writer = csv.DictWriter(stream,fieldnames=['status'])
            writer.writeheader()
            writer.writerow(dict(status=data['summary']['status']))
    print(json.dumps({key:value for key,value in data.items() if key not in ('codes','summary')},indent=2))
    print(json.dumps(data['summary'],indent=2))


if __name__ == '__main__':
    main()
