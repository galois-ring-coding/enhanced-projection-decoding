"""Audit fixed-rate Lee measurements and join the archived n=64 cohort."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from analyze_lee_extended import write_csv
from analyze_lee_fixed import METRICS
from lee_extended import composition_probability, distance_certificate, prepare
from lee_fixed_kernel import decode
from lee_revision import make_code
from lee_z4_dcc_comparison import lee, raw_syndrome
from present_lee_fixed import COLORS, LABELS, expected_searches, validate_stopping
from run_lee_fixed import coverage_diagnostics
from run_lee_independent import ROOT, METHODS, digest, matrix_digest
from run_lee_scale_confirmation import error_pool, search_seed, seed_for


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_rows(folder):
    return [json.loads(line) for line in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]


def audit(folder, rows):
    config = read_json(folder/'config.json')
    n, k, t = (config[v] for v in ('n', 'K', 't'))
    for name, expected in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Measured source changed: '+name)
    codes = read_json(folder/'codes.json')
    by_code = {c['code']: c for c in codes}
    mothers = {c['cluster']: c for c in codes}
    pools = read_json(folder/'error_pools.json')
    if len(codes) != config['mothers']*(k+1) or len(by_code) != len(codes) or len(mothers) != config['mothers']:
        raise AssertionError('Unexpected panel size')
    hashes, distances = set(), {}
    for cluster, info in mothers.items():
        fingerprint = matrix_digest(info['ambient'])
        if fingerprint in hashes:
            raise AssertionError('Duplicate mother')
        hashes.add(fingerprint)
        if info['seed'] != seed_for(config['seed'], n, info['mother_index'], 'ambient'):
            raise AssertionError('Mother seed mismatch')
        distance = distance_certificate(info['ambient'][:k])
        if distance != info['d_binary'] or distance <= 4*t:
            raise AssertionError('Distance certification failed')
        distances[cluster] = distance
        base = make_code(info['ambient'], k, 0)
        for k0 in range(k+1):
            child = by_code[f'{cluster}_k0{k0}']
            h = base['dual'][k:]+[[2*x % 4 for x in r] for r in base['dual'][k0:k]]
            if (child['h'] != h or child['ambient'] != info['ambient']
                    or child['ambient_sha256'] != fingerprint or child['d_binary'] != distance):
                raise AssertionError('Subtype construction mismatch')
        if pools[cluster] != error_pool(config['seed'], n, t, info['mother_index']):
            raise AssertionError('Error pool is not reproducible')
    expected = {}
    for code in codes:
        for case in pools[code['cluster']]:
            for index in range(1 if case['population'] == 'uniform' else config['seeds']):
                for method in METHODS:
                    expected[code['code'], case['population'], case['case'], index, method] = case
    seen, outcomes, pairs = set(), Counter(), defaultdict(dict)
    projection_counts = defaultdict(set)
    for row in rows:
        key = tuple(row[v] for v in ('code', 'population', 'case', 'seed_index', 'method'))
        if key in seen or key not in expected:
            raise AssertionError('Unexpected or duplicate record')
        seen.add(key)
        code, case = by_code[row['code']], expected[key]
        for field in ('error', 'a', 's', 'error_seed'):
            if row[field] != case[field]:
                raise AssertionError('Shared input mismatch')
        for field in ('n', 'K', 't', 'k0', 'd_binary', 'cluster'):
            if row[field] != code[field]:
                raise AssertionError('Code metadata mismatch')
        if row['kernel'] != 'common' or lee(row['error']) != t:
            raise AssertionError('Wrong kernel or weight')
        if row['syndrome'] != raw_syndrome(code['h'], row['error']):
            raise AssertionError('Wrong syndrome')
        seed = search_seed(config['seed'], n, code['mother_index'], row['population'], row['case'], row['seed_index'])
        if row['trial_seed'] != seed:
            raise AssertionError('Wrong search seed')
        if len(row['timing_samples']) != config['repeats']:
            raise AssertionError('Missing timing repeat')
        for metric in row['timing_samples'][0]:
            if row[metric] != statistics.median(s[metric] for s in row['timing_samples']):
                raise AssertionError('Wrong timing median')
        outcomes[row['status']] += 1
        if row['status'] == 'complete' and not row['correct']:
            raise AssertionError('Incorrect completed answer')
        pairs[key[:-1]][row['method']] = row
        if row['method'] == 'proj_binary_all':
            projection_counts[row['cluster'], row['population'], row['case'], row['seed_index']].add(row.get('first_trials', 0))
    if seen != set(expected):
        raise AssertionError('Incomplete schedule')
    for pair in pairs.values():
        ep, projection = (pair[m] for m in METHODS)
        fields = {v for r in pair.values() for v in r if v.startswith('second_')}
        if any(ep.get(v, 0) != projection.get(v, 0) for v in fields):
            raise AssertionError('Matched recovery counters differ')
        if ([e for e in ep['trace'] if e['phase'] == 'recovery'] !=
                [e for e in projection['trace'] if e['phase'] == 'recovery']):
            raise AssertionError('Matched recovery traces differ')
        if ep['k0'] == k:
            for field in ('trace', 'first_trials', 'first_candidates', 'arithmetic_proxy'):
                if ep.get(field, 0) != projection.get(field, 0):
                    raise AssertionError('Free boundary mismatch')
    if any(len(v) != 1 for v in projection_counts.values()):
        raise AssertionError('Projection selection counts changed across subtypes')
    result = dict(status='passed' if outcomes == {'complete': len(rows)} else 'censored',
        n=n, K=k, t=t, mothers=len(mothers), codes=len(codes), records=len(rows),
        timing_calls=sum(len(r['timing_samples']) for r in rows), outcomes=dict(outcomes),
        distances=distances, matched_recovery=True, free_boundary_identical=True)
    if result['status'] != 'passed':
        (folder/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        raise AssertionError('Censored costs retained; complete-only comparison is not allowed')
    result['stopping'] = validate_stopping(rows)
    selected = [r for r in rows if r['population'] == 'a1' and r['case'] == 0 and r['seed_index'] == 0
                and (by_code[r['code']]['mother_index'] == 0 or r['k0'] == k//2)]
    prepared = {}
    for row in selected:
        if row['code'] not in prepared:
            prepared[row['code']] = prepare(by_code[row['code']]['h'])
        replay = decode(prepared[row['code']], row['syndrome'], t, row['trial_seed'], row['method'],
                        config['trials'], config['candidates'])
        replay['counters'].update(coverage_diagnostics(replay, row['error']))
        if replay['answer'] != row['error'] or replay['trace'] != row['trace']:
            raise AssertionError('Replay differs')
        if any(row.get(v, 0) != count for v, count in replay['counters'].items()):
            raise AssertionError('Replayed counters differ')
    result['replayed_counted_calls'] = len(selected)
    (folder/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_baseline(folder):
    manifest = read_json(folder/'analysis/manifest.json')
    for name, expected in manifest['input_hashes'].items():
        if digest(folder/name) != expected:
            raise AssertionError('Archived baseline changed: '+name)
    config = read_json(folder/'config.json')
    for name, expected in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Baseline implementation changed')
    check = read_json(folder/'audit.json')
    if check['status'] != 'passed':
        raise AssertionError('Unaudited baseline')
    return load_rows(folder), check


def weighted(rows):
    """Average cases/seeds within each composition before exact-sphere weighting."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['population'] != 'uniform':
            buckets[row['code'], row['method']][row['a']].append(row)
    result = []
    for (code, method), strata in sorted(buckets.items()):
        example = next(iter(strata.values()))[0]
        n, k, t = (example[v] for v in ('n', 'K', 't'))
        if set(strata) != set(range(t//2+1)):
            raise AssertionError('Missing composition')
        if len({len(v) for v in strata.values()}) != 1:
            raise AssertionError('Unbalanced composition sample sizes')
        for group in ('sphere', 'a_positive'):
            required = [a for a in strata if group == 'sphere' or a > 0]
            weights = {a: float(composition_probability(n, t, a)) for a in required}
            mass = sum(weights.values())
            item = {v: example[v] for v in ('code', 'cluster', 'n', 'K', 't', 'k0')}
            item.update(method=method, group=group, sphere_mass=mass)
            for metric in METRICS:
                item[metric] = sum(weights[a]/mass*statistics.mean(r.get(metric, 0) for r in strata[a]) for a in required)
            result.append(item)
    return result


def summarize(points, repetitions):
    output, ratios, calibration = [], [], []
    for n in sorted({r['n'] for r in points}):
        # Preserve the archived n=64 resampling sequence when joining cohorts.
        rng = np.random.default_rng(2026110409 if n == 64 else 2026110429+n)
        data = [r for r in points if r['n'] == n]
        k, t = data[0]['K'], data[0]['t']
        clusters = sorted({r['cluster'] for r in data})
        lookup = {(r['cluster'], r['k0'], r['group'], r['method']): r for r in data}
        count = len(clusters)
        counts = rng.multinomial(count, np.full(count, 1/count), size=repetitions)
        for group in ('sphere', 'a_positive'):
            arrays = {}
            for method in METHODS:
                for metric in METRICS:
                    arrays[method, metric] = np.array([[lookup[c, k0, group, method][metric]
                        for k0 in range(k+1)] for c in clusters])
                for k0 in range(k+1):
                    item = dict(n=n, K=k, t=t, k0=k0, group=group, method=method,
                        extra_fraction=(k-k0)/k, mothers=count)
                    for metric in METRICS:
                        item[metric] = float(arrays[method, metric][:, k0].mean())
                    first, recovery = expected_searches(n, k, t, k0, method, group)
                    item.update(predicted_first_trials=first, predicted_recovery_trials=recovery)
                    for metric in ('online_ns', 'first_trials', 'recovery_trials'):
                        draws = counts @ arrays[method, metric][:, k0]/count
                        low, high = np.quantile(draws, [.025, .975])
                        item[metric+'_low95'], item[metric+'_high95'] = float(low), float(high)
                    output.append(item)
                for metric, stage in (('first_trials', 0), ('recovery_trials', 1)):
                    values = arrays[method, metric]
                    mean, draws = values.mean(axis=0), counts @ values/count
                    width = float(np.quantile(np.max(np.abs(np.log(draws)-np.log(mean)), axis=1), .95))
                    predicted = [expected_searches(n, k, t, k0, method, group)[stage] for k0 in range(k+1)]
                    for k0 in range(k+1):
                        low, high = mean[k0]*np.exp(-width), mean[k0]*np.exp(width)
                        calibration.append(dict(n=n, k0=k0, method=method, group=group, metric=metric,
                            prediction=predicted[k0], observed=float(mean[k0]),
                            simultaneous_low95=float(low), simultaneous_high95=float(high),
                            prediction_inside=bool(low <= predicted[k0] <= high)))
            for metric in ('online_ns', 'first_trials', 'first_ns', 'recovery_ns', 'arithmetic_proxy'):
                a, b = (arrays[m, metric] for m in METHODS)
                ratio, draws = a.mean(axis=0)/b.mean(axis=0), (counts @ a)/(counts @ b)
                width = float(np.quantile(np.max(np.abs(np.log(draws)-np.log(ratio)), axis=1), .95))
                low, high = np.quantile(draws, [.025, .975], axis=0)
                for k0 in range(k+1):
                    slo, shi = ratio[k0]*np.exp(-width), ratio[k0]*np.exp(width)
                    ratios.append(dict(n=n, K=k, t=t, k0=k0, group=group, metric=metric,
                        extra_fraction=(k-k0)/k, ratio=float(ratio[k0]), low95=float(low[k0]), high95=float(high[k0]),
                        simultaneous_low95=float(slo), simultaneous_high95=float(shi),
                        result='EP_lower' if shi < 1 else 'projection_lower' if slo > 1 else 'unresolved'))
    return output, ratios, calibration


def plot_scale(folder, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lengths = sorted({r['n'] for r in table})
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(len(lengths), 2, figsize=(11, 9.6), constrained_layout=True)
    for row_index, n in enumerate(lengths):
        for method, marker in zip(METHODS, ('o', 's')):
            data = sorted((r for r in table if r['n'] == n and r['group'] == 'sphere' and r['method'] == method),
                          key=lambda r: r['extra_fraction'])
            x = [r['extra_fraction'] for r in data]
            axes[row_index, 0].plot(x, [r['predicted_first_trials'] for r in data], '--', color=COLORS[method])
            for col, metric, scale in ((0, 'first_trials', 1), (1, 'online_ns', 1e6)):
                ax = axes[row_index, col]
                mean = np.array([r[metric]/scale for r in data])
                low = np.array([r[metric+'_low95']/scale for r in data])
                high = np.array([r[metric+'_high95']/scale for r in data])
                ax.errorbar(x, mean, yerr=np.array([mean-low, high-mean]), fmt=marker,
                    markersize=4, color=COLORS[method], capsize=2, elinewidth=.7,
                    errorevery=len(x) if metric == 'first_trials' and method == 'proj_binary_all' else 1)
                if col == 1 or method == 'proj_binary_all':
                    ax.plot(x, mean, color=COLORS[method], linewidth=1)
        k, t = data[0]['K'], data[0]['t']
        axes[row_index, 0].set_title(f'n={n}, K={k}, t={t}: initial selections')
        axes[row_index, 1].set_title(f'n={n}, K={k}, t={t}: online cost')
        axes[row_index, 0].set_ylabel('Mean first-stage selections')
        axes[row_index, 1].set_ylabel('Mean online time (ms)')
        for ax in axes[row_index]:
            ax.set_xlim(-.035, 1.035)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', color='.9', linewidth=.6)
            ax.set_xlabel(r'Additional-constraint fraction $(K-k_0)/K$')
    handles = [Line2D([0], [0], color=COLORS[m], marker=marker, label=LABELS[m])
               for m, marker in zip(METHODS, ('o', 's'))]
    axes[0, 0].legend(handles=handles+[
        Line2D([0], [0], color='.4', linestyle='--', label='Exact coverage prediction')],
        loc='lower left', fontsize=8, frameon=False)
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'scale_search_to_cost.{extension}', dpi=220)
    plt.close(fig)


def stage_table(table):
    lookup = {(r['n'], r['k0'], r['method'], r['group']): r for r in table}
    result = []
    for n in sorted({r['n'] for r in table}):
        k = next(r['K'] for r in table if r['n'] == n)
        for k0 in sorted({0, k//3, k//2, 2*k//3, k}):
            for method in METHODS:
                row, high = (lookup[n, k0, method, g] for g in ('sphere', 'a_positive'))
                result.append(dict(n=n, K=k, t=row['t'], k0=k0, method=LABELS[method],
                    extra_fraction=(k-k0)/k, predicted_first_trials=row['predicted_first_trials'],
                    measured_first_trials=row['first_trials'], first_ms=row['first_ns']/1e6,
                    recovery_ms=row['recovery_ns']/1e6, online_ms=row['online_ns']/1e6,
                    arithmetic_proxy_million=row['arithmetic_proxy']/1e6,
                    first_linear_ms_per_trial=row['first_linear_ns']/row['first_trials']/1e6,
                    high_group_online_ms=high['online_ns']/1e6,
                    high_group_recovery_ms=high['recovery_ns']/1e6))
    return result


def mother_ratios(points):
    lookup = {(r['n'], r['cluster'], r['k0'], r['method'], r['group']): r for r in points}
    result = []
    for row in points:
        if row['method'] != 'ep_all':
            continue
        other = lookup[row['n'], row['cluster'], row['k0'], 'proj_binary_all', row['group']]
        result.append(dict(n=row['n'], K=row['K'], t=row['t'], cluster=row['cluster'],
            k0=row['k0'], group=row['group'], online_ratio=row['online_ns']/other['online_ns'],
            first_trial_ratio=row['first_trials']/other['first_trials'],
            arithmetic_ratio=row['arithmetic_proxy']/other['arithmetic_proxy']))
    return result


def uniform_summary(rows):
    buckets = defaultdict(list)
    for row in rows:
        if row['population'] == 'uniform':
            buckets[row['n'], row['k0'], row['method']].append(row)
    return [dict(n=n, k0=k0, method=method, calls=len(data), correct=sum(r['correct'] for r in data),
        online_ms=statistics.mean(r['online_ns'] for r in data)/1e6,
        first_trials=statistics.mean(r.get('first_trials', 0) for r in data),
        recovery_trials=statistics.mean(r['recovery_trials'] for r in data))
        for (n, k0, method), data in sorted(buckets.items())]


def write_report(folder, table, ratios, calibration, checks, stages):
    lines = ['# 固定比例的 Lee 译码规模实验', '',
        '比较Z4上的EP与普通投影。双方使用相同受限RREF、枚举一致奇异系统、首阶段半径t和条件二元高位恢复。'
        '本实验中的普通投影是匹配二元恢复基准。', '',
        '## 设计与测量', '',
        '固定K/n=3/16、t/n=1/16，统一按d_H(D1)>4t筛选母矩阵。每组16个独立母矩阵，'
        '遍历k0=0,...,K。新增n=48和96，n=64复用已经核查并保留哈希的独立码实验。', '',
        '| n | K | t | 子型码数 | 距离范围 | 计数调用 | 计时调用 |',
        '|---:|---:|---:|---:|---|---:|---:|']
    for check in checks:
        distances = list(check.get('distances', check.get('distance_certificates', {})).values())
        lines.append(f'| {check["n"]} | {check["K"]} | {check["t"]} | {check["codes"]} | '
            f'{min(distances)}--{max(distances)} | {check["records"]:,} | {check["timing_calls"]:,} |')
    lines += ['', 'a表示符号2的数量，奇数位置数为s=t-2a。每个母矩阵、每种组成各取2个独立错误，'
        '每个错误用4个搜索种子；另取2个均匀精确Lee球面错误，各用1个种子，单独验证。'
        '同一母矩阵下，错误与搜索种子在全部子型和双方间共享。计数调用后做3次不记录计数/轨迹的计时，取中位数。', '',
        '组成概率为', '',
        '$$w_a=\\frac{\\binom{n}{t-2a}\\binom{n-t+2a}{a}2^{t-2a}}{\\binom{2n}{t}}.$$', '',
        '先在每种组成内部平均错误和搜索种子，再按w_a加权。含2子组只保留a>0并重新归一化。'
        '均匀验证样本不混入主估计。', '',
        '在线时间包括系统构造、低位搜索、高位恢复和最终验证，排除码生成、距离认证和缓存Smith预处理。'
        '三组测量来自各自批次，比较不同规模时应结合各批次内部的EP/投影比值。', '',
        '## 主图', '', '![规模实验](scale_search_to_cost.png)', '',
        '三行分别是n=48、64、96。横轴(K-k0)/K从0到1，表示额外约束占K的比例；'
        '0对应自由码边界k0=K，1对应k0=0。蓝色EP、红色普通投影。', '',
        '左列虚线是精确覆盖模型的首阶段平均选集次数，点为实测均值。右列为完整在线时间。'
        '误差棒按母矩阵重抽样，为逐点95%区间。普通投影的选集次数在同一长度的各子型上逐输入相同，'
        '左图只在横轴0处画一次公共误差棒。', '',
        '当s>0时，选集覆盖概率为binom(n-k0,s)/binom(n,s)和binom(n-K,s)/binom(n,s)。'
        '当前唯一性与枚举策略下，每次非零搜索均在第一次覆盖真实支撑时成功，平均次数为1/P。'
        's=0走零伴随式捷径，计0次。高位a>0时的平均次数为binom(n-s,a)/binom(n-K-s,a)，a=0直接恢复。', '',
        '## 配对时间比与子型判断', '',
        '下表依据EP/普通投影的均值时间比。按每个长度、分组、指标分别构造同时覆盖全部子型的95% bootstrap区间；'
        '没有通过两条误差棒是否重叠来判断优势。', '',
        '| n | 错误组 | EP时间更低的k0 | 普通投影时间更低的k0 | 区间包含1的k0 |',
        '|---:|---|---|---|---|']
    for n in sorted({r['n'] for r in table}):
        for group in ('sphere', 'a_positive'):
            cells = [r for r in ratios if r['n'] == n and r['group'] == group and r['metric'] == 'online_ns']
            selections = [', '.join(str(r['k0']) for r in cells if r['result'] == flag) or '无'
                          for flag in ('EP_lower', 'projection_lower', 'unresolved')]
            lines.append(f'| {n} | {group} | '+' | '.join(selections)+' |')
    lines += ['', '整球估计和含2子组分别报告。各组成中低位支撑大小不同，低位覆盖收益会变化；'
        '匹配双方的高位搜索轨迹相同，完整成本还包括共同的恢复和验证。', '',
        '以下列出纯torsion、半维自由部分和自由码边界的时间比。n=48的中间点为k0=4，'
        '其自由部分占比4/9；另外两组的中间点恰好为1/2。', '',
        '| n | k0 | EP在线ms | 投影在线ms | 时间比 | 同时95%区间 |',
        '|---:|---:|---:|---:|---:|---|']
    lookup = {(r['n'], r['k0'], r['group'], r['method']): r for r in table}
    for row in ratios:
        if row['metric'] != 'online_ns' or row['group'] != 'sphere' or row['k0'] not in (0, row['K']//2, row['K']):
            continue
        ep, projection = (lookup[row['n'], row['k0'], 'sphere', m] for m in METHODS)
        lines.append(f'| {row["n"]} | {row["k0"]} | {ep["online_ns"]/1e6:.3f} | '
            f'{projection["online_ns"]/1e6:.3f} | {row["ratio"]:.3f} | '
            f'[{row["simultaneous_low95"]:.3f}, {row["simultaneous_high95"]:.3f}] |')
    lines += ['', '## 阶段成本与高位子组', '',
        'stage_costs.csv/.tex列出每个长度的k0=0、floor(K/3)、floor(K/2)、floor(2K/3)、K。'
        '字段包含低位次数、低位/恢复/完整时间、单次低位消元时间、算术代理和含2子组完整时间。'
        '每种方法的阶段时间各自取中位数，阶段之和无需严格等于完整在线时间。', '',
        '| n | k0 | 方法 | 首阶段次数 | 低位ms | 恢复ms | 在线ms | 含2组在线ms |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in stages:
        if row['k0'] == row['K']//2:
            lines.append(f'| {row["n"]} | {row["k0"]} | {row["method"]} | {row["measured_first_trials"]:.3f} | '
                f'{row["first_ms"]:.3f} | {row["recovery_ms"]:.3f} | {row["online_ms"]:.3f} | '
                f'{row["high_group_online_ms"]:.3f} |')
    new_checks = [r for r in checks if r['n'] != 64]
    lines += ['', '## 核查与复现', '',
        f'- 新增{sum(r["records"] for r in new_checks):,}次计数调用和'
        f'{sum(r["timing_calls"] for r in new_checks):,}次计时调用；全部正确完成，预算截断为0。',
        f'- 对新组另重放{sum(r["replayed_counted_calls"] for r in new_checks)}次，答案、轨迹、计数一致。',
        '- 新组逐一重新检查母矩阵可逆性、距离、子型校验矩阵、输入伴随式、种子和计时中位数。',
        '- 每条非零搜索都在第一次覆盖真实支撑时停止；双方高位轨迹/计数逐输入一致，自由码边界搜索计数一致。',
        f'- 覆盖预测校准：{len(calibration)}个预测中，'
        f'{sum(r["prediction_inside"] for r in calibration)}个位于对应观测均值的同时区间内。',
        '- n=64原始记录与既有核查报告的哈希一致；此次未重新计时。', '',
    ]
    failed = defaultdict(list)
    for row in calibration:
        if not row['prediction_inside']:
            failed[row['n'], row['method'], row['group'], row['metric']].append(row)
    for (n, method, group, metric), cells in sorted(failed.items()):
        example = cells[0]
        lines += [f'覆盖校准中，n={n}、{LABELS[method]}、{group}、{metric}：'
            f'预测均值{example["prediction"]:.3f}，实测{example["observed"]:.3f}，'
            f'观测均值的同时95%区间为[{example["simultaneous_low95"]:.3f}, '
            f'{example["simultaneous_high95"]:.3f}]。'
            f'共有{len(cells)}个子型行未覆盖预测。普通投影各子型共享同一选集次数，'
            '这些行对应同一条观测曲线。该轮普通投影偏高的选集等待次数也影响完整时间差。', '']
    lines += ['文件：scale_search_to_cost.png/.pdf为主图；panel_summary.csv保存全部均值；paired_ratios.csv保存比值和区间；'
        'prediction_check.csv保存覆盖校准；weighted_by_code.csv和mother_ratios.csv保存逐母矩阵统计；'
        'composition_weights.csv保存各码长的精确组成概率；uniform_validation.csv保存均匀球面验证；'
        'manifest.json记录全部输入、代码和输出哈希。', '',
        '```text', 'python src/run_lee_scale_confirmation.py',
        'python src/analyze_lee_scale_confirmation.py', '```', '']
    (folder/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    tex = [r'\begin{tabular}{rrlrrrr}', r'\toprule',
        r'$n$ & $k_0$ & Method & Low (ms) & Recovery (ms) & Online (ms) & Online, $a>0$ (ms) \\', r'\midrule']
    for row in stages:
        tex.append(f'{row["n"]} & {row["k0"]} & {row["method"]} & {row["first_ms"]:.3f} & '
            f'{row["recovery_ms"]:.3f} & {row["online_ms"]:.3f} & {row["high_group_online_ms"]:.3f} '+r'\\')
    (folder/'stage_costs.tex').write_text('\n'.join(tex+[r'\bottomrule', r'\end{tabular}', '']), encoding='utf-8')
    (folder/'caption.tex').write_text(
        'Fixed-rate Lee decoding over $\\mathbb Z_4$ with $K/n=3/16$ and $t/n=1/16$. '
        'Rows show $(n,K,t)=(48,9,3),(64,12,4),(96,18,6)$, each using 16 mother matrices '
        'conditioned on $d_H(D_1)>4t$. Both methods use matched all-solutions binary searches '
        'and conditional recovery. Left: exact coverage predictions (dashed) and measured '
        'first-stage selections (markers). Right: complete online time excluding cached preprocessing. '
        'All subtypes use two errors per composition and four search seeds per error, weighted by '
        'exact Lee-sphere composition probabilities. Error bars are pointwise 95\\% mother-cluster '
        'bootstrap intervals. The $n=64$ measurements are reused from the independent-code cohort.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/lee_scale_confirmation_20261104')
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/lee_independent_20261104')
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('At least two bootstrap replications are required')
    baseline, previous = load_baseline(args.baseline)
    previous = dict(previous, n=64, K=12, t=4)
    rows, checks = list(baseline), [previous]
    inputs = [args.baseline]
    for n in (48, 96):
        source = args.source/f'n{n}'
        config = read_json(source/'config.json')
        if (config['mothers'], config['seeds'], config['repeats']) != (16, 4, 3):
            raise AssertionError('Formal presentation requires the full planned design')
        measured = load_rows(source)
        check = audit(source, measured)
        rows.extend(measured)
        checks.append(check)
        inputs.append(source)
        print(json.dumps(check), flush=True)
    points = weighted(rows)
    table, ratios, calibration = summarize(points, args.bootstrap)
    stages = stage_table(table)
    output = args.source/'analysis'
    output.mkdir(exist_ok=True)
    compositions = [dict(n=n, K=k, t=t, a=a, s=t-2*a,
        probability_fraction=str(composition_probability(n, t, a)),
        probability=float(composition_probability(n, t, a)))
        for n, k, t in sorted({(r['n'], r['K'], r['t']) for r in table}) for a in range(t//2+1)]
    for name, data in (('weighted_by_code', points), ('panel_summary', table),
            ('paired_ratios', ratios), ('prediction_check', calibration),
            ('mother_ratios', mother_ratios(points)), ('stage_costs', stages),
            ('uniform_validation', uniform_summary(rows)), ('composition_weights', compositions)):
        write_csv(output/(name+'.csv'), data)
    checks.sort(key=lambda r: r['n'])
    (output/'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    plot_scale(output, table)
    write_report(output, table, ratios, calibration, checks, stages)
    manifest = dict(bootstrap=args.bootstrap,
        input_hashes={str(folder.resolve()): {name: digest(folder/name) for name in
            ('config.json', 'codes.json', 'error_pools.json', 'instances.jsonl', 'audit.json', 'environment.json')}
            for folder in inputs},
        analysis_sources={name: digest(ROOT/'src'/name) for name in
            ('analyze_lee_scale_confirmation.py', 'analyze_lee_fixed.py', 'analyze_lee_extended.py', 'present_lee_fixed.py')},
        output_hashes={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps([r for r in ratios if r['metric'] == 'online_ns' and r['group'] == 'sphere'
                      and r['k0'] in (0, r['K']//2, r['K'])]), flush=True)


if __name__ == '__main__':
    main()
