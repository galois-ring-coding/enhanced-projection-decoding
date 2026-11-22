"""Audit and analyze fixed-input Lee kernel/seed diagnostics, without new timings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from analyze_lee_extended import write_csv
from lee_extended import composition_probability, prepare
from lee_fixed_kernel import MATCHED, decode
from run_lee_fixed import SOURCES, schedule

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('sphere', 'a_zero', 'a_positive', 'a_two')
METRICS = ('online_ns', 'first_ns', 'recovery_ns', 'verify_ns',
           'first_linear_ns', 'recovery_linear_ns', 'known_elimination_ns',
           'first_trials', 'recovery_trials', 'first_candidates', 'second_candidates',
           'arithmetic_proxy', 'first_arithmetic_proxy', 'second_arithmetic_proxy',
           'first_singular_trials', 'first_consistent_singular_trials',
           'first_inconsistent_trials', 'first_full_rank_trials',
           'first_covered_trials', 'first_covered_singular_trials',
           'second_singular_trials', 'second_inconsistent_trials',
           'first_rank_defect_0', 'first_rank_defect_1', 'first_rank_defect_2',
           'first_rank_defect_3', 'first_rank_defect_4')
COMPARISONS = (
    ('common_reject_EP_vs_projection', ('common', 'ep_reject'), ('common', 'proj_binary_t'), 'all'),
    ('common_enumerate_EP_vs_projection', ('common', 'ep_all'), ('common', 'proj_binary_all'), 'all'),
    ('common_EP_enumerate_vs_reject', ('common', 'ep_all'), ('common', 'ep_reject'), 'all'),
    ('common_projection_enumerate_vs_reject', ('common', 'proj_binary_all'), ('common', 'proj_binary_t'), 'all'),
    ('legacy_reject_EP_vs_projection', ('legacy', 'ep_reject'), ('legacy', 'proj_binary_t'), 'original'),
    ('legacy_enumerate_EP_vs_projection', ('legacy', 'ep_all'), ('legacy', 'proj_binary_all'), 'original'),
    ('common_vs_legacy_EP_reject', ('common', 'ep_reject'), ('legacy', 'ep_reject'), 'original'),
    ('common_vs_legacy_projection_reject', ('common', 'proj_binary_t'), ('legacy', 'proj_binary_t'), 'original'),
    ('common_vs_legacy_EP_enumerate', ('common', 'ep_all'), ('legacy', 'ep_all'), 'original'),
    ('common_vs_legacy_projection_enumerate', ('common', 'proj_binary_all'), ('legacy', 'proj_binary_all'), 'original'),
    ('common_EP_vs_legacy_ring', ('common', 'ep_all'), ('legacy', 'proj_ring_t'), 'original'),
)


def accepts(group, a):
    return group == 'sphere' or (group == 'a_zero' and a == 0) or (
        group == 'a_positive' and a > 0) or (group == 'a_two' and a == 2)


def audit(folder, rows):
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    run = json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    source = Path(config['source'])
    for name in SOURCES:
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() != config['source_hashes'][name]:
            raise AssertionError('Measured source changed: '+name)
    for name, digest in config['input_hashes'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise AssertionError('Archived input changed: '+name)
    codes = {c['code']: c for c in json.loads((source/'codes.json').read_text(encoding='utf-8'))
             if (c['n'], c['K'], c['t']) == (64, 12, 4)}
    originals = {}
    with (source/'instances.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['code'] in codes and row['method'] == 'ep_all':
                originals[row['code'], row['population'], row['case']] = row
    actual, groups, by_method = {}, defaultdict(dict), defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('code', 'population', 'case', 'seed_index', 'kernel', 'method'))
        if key in actual:
            raise AssertionError('Duplicate measured row')
        actual[key] = row
        original = originals[key[:3]]
        for field in ('error', 'syndrome', 'a', 's', 'd_binary', 'cluster'):
            if row[field] != original[field]:
                raise AssertionError('Changed archived input: '+field)
        if len(row['timing_samples']) != config['repeats']:
            raise AssertionError('Missing timing samples')
        if row['status'] != 'complete' or not row['correct']:
            raise AssertionError('Censored/incorrect run; do not report complete cost')
        if row['kernel'] == 'common':
            for phase, prefix in (('first', 'first'), ('recovery', 'second')):
                trace = [event for event in row['trace'] if event['phase'] == phase]
                if len(trace) != row.get(prefix+'_trials', 0):
                    raise AssertionError('Trace/trial mismatch')
                if sum(event['nullity'] > 0 for event in trace) != row.get(prefix+'_singular_trials', 0):
                    raise AssertionError('Trace/singular mismatch')
                if sum(not event['consistent'] for event in trace) != row.get(prefix+'_inconsistent_trials', 0):
                    raise AssertionError('Trace/consistency mismatch')
                support = ({j for j, v in enumerate(row['error']) if v % 2} if phase == 'first' else
                           {i for i, j in enumerate(j for j, v in enumerate(row['error']) if v % 2 == 0)
                            if row['error'][j] == 2})
                for event in trace:
                    if event['covers_true_support'] != support.issubset(event['chosen']):
                        raise AssertionError('Post-hoc coverage mismatch')
                if row.get(phase+'_covered_trials', 0) != sum(e['covers_true_support'] for e in trace):
                    raise AssertionError('Post-hoc coverage count mismatch')
            groups[key[:4]][row['method']] = row
        by_method[row['kernel'], row['method']].append(row)
    expected = set()
    measured_codes = {r['code'] for r in rows}
    if len(measured_codes) != run['codes']:
        raise AssertionError('Missing measured code')
    for original in originals.values():
        if original['code'] not in measured_codes:
            continue
        for index, trial_seed, kernel, methods in schedule(original, config['seeds'], config['focus_seeds']):
            for method in methods:
                key = original['code'], original['population'], original['case'], index, kernel, method
                expected.add(key)
                if key in actual and actual[key]['trial_seed'] != trial_seed:
                    raise AssertionError('Search seed mismatch')
    if set(actual) != expected or len(actual) != run['records']:
        raise AssertionError('Incomplete measurement panel')
    for pair in groups.values():
        if set(pair) != set(MATCHED):
            raise AssertionError('Unpaired common methods')
        for ep, projection in (('ep_all', 'proj_binary_all'), ('ep_reject', 'proj_binary_t')):
            left, right = pair[ep], pair[projection]
            for field in set(left) | set(right):
                if field.startswith('second_') and left.get(field, 0) != right.get(field, 0):
                    raise AssertionError('Matched recovery counters differ')
            if left['k0'] == 12:
                for field in ('first_trials', 'first_candidates', 'arithmetic_proxy', 'trace'):
                    if left.get(field, 0) != right.get(field, 0):
                        raise AssertionError('Free-boundary mismatch: '+field)
        for method in ('ep_reject', 'proj_binary_t'):
            enum = pair['ep_all' if method == 'ep_reject' else 'proj_binary_all']
            reject = pair[method]
            for phase in ('first', 'recovery'):
                etrace = [e for e in enum['trace'] if e['phase'] == phase]
                rtrace = [e for e in reject['trace'] if e['phase'] == phase]
                if etrace != rtrace[:len(etrace)]:
                    raise AssertionError('Policies use different selected systems')
    replays = 0
    for (kernel, method), data in by_method.items():
        for index in sorted({0, len(data)//2, len(data)-1}):
            row = data[index]
            result = decode(prepare(codes[row['code']]['h']), row['syndrome'], 4,
                            row['trial_seed'], method, config['trials'], config['candidates'], kernel=kernel)
            if result['status'] != row['status'] or result['answer'] != row['error']:
                raise AssertionError('Replay answer mismatch')
            for field, value in result['counters'].items():
                if row.get(field, 0) != value:
                    raise AssertionError('Replay counters differ: '+field)
            replays += 1
    result = dict(status='passed', records=len(rows), timing_calls=len(rows)*config['repeats'],
                  codes=len({r['code'] for r in rows}), mothers=len({r['cluster'] for r in rows}),
                  archived_inputs=len({(r['code'], r['population'], r['case']) for r in rows}),
                  common_seed_input_pairs=len(groups), new_codes=0, new_errors=0,
                  censored=0, incorrect=0, replayed_calls=replays,
                  verified='Source/input hashes, full schedule, trace ranks/consistency/coverage, '
                           'shared policy selection prefixes, matched recovery, free boundary, deterministic replays.')
    (folder/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def weighted(rows):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['population'] == 'uniform':
            continue
        for scope in ('all', 'original') if row['seed_index'] == 0 else ('all',):
            buckets[row['code'], row['kernel'], row['method'], scope][row['a']].append(row)
    result = []
    for (code, kernel, method, scope), strata in sorted(buckets.items()):
        example = next(iter(strata.values()))[0]
        if kernel == 'legacy' and scope == 'all':
            continue
        for group in GROUPS:
            required = [a for a in range(3) if accepts(group, a)]
            if set(required)-set(strata):
                raise AssertionError('Missing error compositions')
            mass = sum(float(composition_probability(64, 4, a)) for a in required)
            item = {k: example[k] for k in ('code', 'cluster', 'k0')}
            item.update(kernel=kernel, method=method, seed_scope=scope, group=group, sphere_mass=mass)
            for metric in METRICS:
                item[metric] = sum(float(composition_probability(64, 4, a))/mass *
                                   statistics.mean(r.get(metric, 0) for r in strata[a]) for a in required)
            result.append(item)
    return result


def summaries(points, repetitions):
    families = defaultdict(list)
    for row in points:
        families[row['kernel'], row['method'], row['seed_scope'], row['group'], row['k0']].append(row)
    result = []
    rng = np.random.default_rng(2026110403)
    for (kernel, method, scope, group, k0), data in sorted(families.items()):
        item = dict(kernel=kernel, method=method, seed_scope=scope, group=group, k0=k0, mothers=len(data))
        for metric in METRICS:
            item[metric] = statistics.mean(r[metric] for r in data)
        counts = rng.multinomial(len(data), np.full(len(data), 1/len(data)), size=repetitions)
        draws = counts @ np.array([r['online_ns'] for r in data])/len(data)
        item['online_low95'], item['online_high95'] = np.quantile(draws, [.025, .975])
        trials = item['first_trials']
        item['first_linear_ns_per_trial'] = item['first_linear_ns']/trials if trials else 0
        item['first_singular_fraction'] = item['first_singular_trials']/trials if trials else 0
        item['first_coverage_fraction'] = item['first_covered_trials']/trials if trials else 0
        item['first_covered_singular_fraction'] = item['first_covered_singular_trials']/trials if trials else 0
        result.append(item)
    return result


def comparisons(points, repetitions):
    lookup = {(r['kernel'], r['method'], r['seed_scope'], r['group'], r['k0'], r['cluster']): r
              for r in points}
    result = []
    for label, numerator, denominator, scope in COMPARISONS:
        for group in GROUPS:
            for metric in ('online_ns', 'first_trials', 'arithmetic_proxy'):
                cells = []
                for k0 in sorted({r['k0'] for r in points}):
                    clusters = sorted({r['cluster'] for r in points if r['k0'] == k0})
                    left = [lookup.get((*numerator, scope, group, k0, c)) for c in clusters]
                    right = [lookup.get((*denominator, scope, group, k0, c)) for c in clusters]
                    if any(r is None for r in left+right):
                        continue
                    a = np.array([r[metric] for r in left])
                    b = np.array([r[metric] for r in right])
                    if not a.mean() > 0 or not b.mean() > 0:
                        continue
                    cells.append((k0, clusters, a, b))
                if not cells:
                    continue
                if any(c[1] != cells[0][1] for c in cells):
                    raise AssertionError('Unequal mother panel')
                n = len(cells[0][1])
                a = np.column_stack([c[2] for c in cells])
                b = np.column_stack([c[3] for c in cells])
                ratio = a.mean(axis=0)/b.mean(axis=0)
                rng = np.random.default_rng(2026110417)
                counts = rng.multinomial(n, np.full(n, 1/n), size=repetitions)
                draws = (counts @ a)/(counts @ b)
                lower, upper = np.quantile(draws, [.025, .975], axis=0)
                width = np.quantile(np.max(np.abs(np.log(draws)-np.log(ratio)), axis=1), .95)
                for i, (k0, _, _, _) in enumerate(cells):
                    low, high = ratio[i]*np.exp(-width), ratio[i]*np.exp(width)
                    result.append(dict(comparison=label, group=group, metric=metric, k0=k0,
                        seed_scope=scope, mothers=n, family_points=len(cells),
                        numerator_mean=a[:, i].mean(), denominator_mean=b[:, i].mean(), ratio=ratio[i],
                        low95=lower[i], high95=upper[i], simultaneous_low95=low, simultaneous_high95=high,
                        result='numerator_lower' if high < 1 else 'denominator_lower' if low > 1 else 'unresolved'))
    return result


def seed_variability(rows):
    buckets = defaultdict(list)
    for row in rows:
        if row['kernel'] == 'common' and row['population'] != 'uniform':
            buckets[row['code'], row['population'], row['case'], row['method']].append(row)
    result = []
    for (code, population, case, method), data in sorted(buckets.items()):
        item = dict(code=code, cluster=data[0]['cluster'], k0=data[0]['k0'], a=data[0]['a'],
                    population=population, case=case, method=method, seeds=len(data))
        for metric in ('first_trials', 'recovery_trials', 'online_ns', 'first_ns'):
            values = [r.get(metric, 0) for r in data]
            item[metric+'_mean'] = statistics.mean(values)
            item[metric+'_sd'] = statistics.stdev(values) if len(values) > 1 else 0
            item[metric+'_min'], item[metric+'_max'] = min(values), max(values)
            item[metric+'_original'] = next(r.get(metric, 0) for r in data if r['seed_index'] == 0)
        result.append(item)
    return result


def coverage_predictions(table):
    result = []
    for row in table:
        if row['kernel'] != 'common' or row['seed_scope'] != 'all' or not row['method'].endswith('_all'):
            continue
        k0 = row['k0'] if row['method'] == 'ep_all' else 12
        required = [a for a in range(3) if accepts(row['group'], a)]
        mass = sum(float(composition_probability(64, 4, a)) for a in required)
        first, recovery = 0, 0
        for a in required:
            s = 4-2*a
            probability = float(composition_probability(64, 4, a))/mass
            first += probability*(math.comb(64, s)/math.comb(64-k0, s) if s else 0)
            recovery += probability*(math.comb(64-s, a)/math.comb(52-s, a) if a else 0)
        result.append(dict(k0=row['k0'], method=row['method'], group=row['group'],
                           predicted_first_trials=first, measured_first_trials=row['first_trials'],
                           predicted_recovery_trials=recovery, measured_recovery_trials=row['recovery_trials']))
    return result


def plots(folder, table, ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    lookup = {(r['kernel'], r['method'], r['seed_scope'], r['group'], r['k0']): r for r in table}
    ks = sorted({r['k0'] for r in table})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey='row', constrained_layout=True)
    for col, (title, ep, proj) in enumerate((('Reject singular', 'ep_reject', 'proj_binary_t'),
                                           ('Enumerate consistent singular', 'ep_all', 'proj_binary_all'))):
        for method, label, color, style in ((ep, 'EP', '#1b789d', '-o'),
                                             (proj, 'Ordinary projection', '#c34f40', '--s')):
            data = [lookup['common', method, 'all', 'sphere', k] for k in ks]
            axes[0, col].plot(ks, [r['online_ns']/1e6 for r in data], style, color=color, label=label)
            axes[0, col].fill_between(ks, [r['online_low95']/1e6 for r in data],
                                     [r['online_high95']/1e6 for r in data], color=color, alpha=.12)
            axes[1, col].plot(ks, [r['first_trials'] for r in data], style, color=color, label=label)
        axes[0, col].set_title(title)
        axes[0, col].legend()
        axes[0, col].set_ylabel('Mean online time (ms)')
        axes[1, col].set_ylabel('Mean first-stage selections')
        axes[1, col].set_xlabel(r'Subtype parameter $k_0$')
        for ax in axes[:, col]:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    fig.suptitle('Z4 Lee: common restricted-matrix elimination; n=64, K=12, t=4\n'
                 'Existing 16 mothers/errors; 4 seeds (8 at k0=4,5,6,7,8,12); exact-sphere weights')
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'common_kernel_comparison.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, constrained_layout=True)
    for ax, method, title in zip(axes, ('ep_reject', 'proj_binary_t'), ('EP / reject', 'Projection / reject')):
        for scope, label, color, style in (('original', 'Original search seed', '#c34f40', '--s'),
                                            ('all', 'Mean over 4 / 8 search seeds', '#1b789d', '-o')):
            data = [lookup['common', method, scope, 'sphere', k] for k in ks]
            ax.plot(ks, [r['first_trials'] for r in data], style, label=label, color=color)
        ax.set_title(title)
        ax.set_xlabel(r'Subtype parameter $k_0$')
        ax.set_ylabel('Mean first-stage selections')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Same codes and errors: sensitivity to random selection seeds')
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'seed_sensitivity.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for ax, group, title in zip(axes, ('sphere', 'a_positive'),
                                ('Uniform exact Lee sphere', 'Conditional on at least one symbol 2')):
        for label, text, color, style in (
                ('common_reject_EP_vs_projection', 'Reject singular', '#c34f40', '--s'),
                ('common_enumerate_EP_vs_projection', 'Enumerate consistent singular', '#1b789d', '-o')):
            data = [r for r in ratios if r['comparison'] == label and r['group'] == group
                    and r['metric'] == 'online_ns']
            data.sort(key=lambda r: r['k0'])
            ax.plot([r['k0'] for r in data], [r['ratio'] for r in data], style, label=text, color=color)
            ax.fill_between([r['k0'] for r in data], [r['simultaneous_low95'] for r in data],
                            [r['simultaneous_high95'] for r in data], color=color, alpha=.12)
        ax.axhline(1, color='.4', lw=1)
        ax.set_title(title)
        ax.set_xlabel(r'Subtype parameter $k_0$')
        ax.set_ylabel('Mean EP time / mean projection time')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Common kernel: paired mother-cluster bootstrap; simultaneous 95% bands over 13 subtypes')
    for ext in ('png', 'pdf'):
        fig.savefig(folder/f'paired_time_ratios.{ext}', dpi=180)
    plt.close(fig)


def report(folder, table, ratios, audit_result):
    lookup = {(r['kernel'], r['method'], r['seed_scope'], r['group'], r['k0']): r for r in table}
    lines = ['# 固定 Z4 Lee 参数的译码核与搜索种子诊断', '',
        '参数固定为 n=64、K=12、t=4。全部复用原 16 个母矩阵、13 个子型以及每个码的原有错误；没有新生成码或错误。继承原 d_H(D1)>16 的筛选条件。', '',
        '## 实验设计', '',
        '* 每个分层错误使用 4 个搜索种子；k0=4,5,6,7,8,12 使用 8 个。种子 0 保留原轨迹。均匀附加样本仅用种子 0 做验证，不混入组成加权均值。',
        '* common 四方法共用选中列上的完整二元消元；唯一分支是奇异且一致时拒绝还是流式枚举。已知支撑消去、半径、恢复和验证规则相同。',
        '* legacy 五方法只在种子 0 重新计时，供原优化实现比较；旧实验文件未修改，不拼接旧计时。',
        '* 每条轨迹先计数，再无计数运行三次取时间中位数。计时顺序轮换。在线时间含构造、搜索、恢复和验证，不包含预处理。',
        '* 先在每个码/组成内平均搜索种子，再按精确球面组成概率加权，最后平均母矩阵。母矩阵簇 bootstrap 5000 次；同时带每次覆盖一个比较、一个分组、一个指标下的全部 13 个子型。',
        '* 本轮错误与码固定，多种子增加的是搜索随机性的观测，不增加独立码或独立错误样本。', '',
        '## 执行核查', '',
        f'* 共 {audit_result["records"]:,} 次计数调用，{audit_result["timing_calls"]:,} 次计时调用；全部恢复原错误，无预算截断。',
        '* 种子 0 的试探/候选计数与归档数据一致；匹配方法恢复计数一致；k0=12 的首阶段计数和整个选集轨迹一致。',
        '* 拒绝与枚举策略在停止前使用同一选集序列；每个选集的秩、一致性及事后支撑覆盖已核查。',
        f'* 另重放 {audit_result["replayed_calls"]} 次调用并比较确定性计数。audit.json 保存核查范围。', '']
    if all(('common', 'ep_reject', 'all', 'sphere', k) in lookup for k in (5, 6, 12)):
        original = lookup['common', 'ep_reject', 'original', 'sphere', 5]
        multiple = lookup['common', 'ep_reject', 'all', 'sphere', 5]
        ep = lookup['common', 'ep_reject', 'all', 'sphere', 6]
        projection = lookup['common', 'proj_binary_t', 'all', 'sphere', 6]
        ep_all = lookup['common', 'ep_all', 'all', 'sphere', 6]
        projection_all = lookup['common', 'proj_binary_all', 'all', 'sphere', 6]
        lines += ['## 本轮回答了什么', '',
            f'1. 原 k0=5 的尖峰对搜索种子敏感：EP拒绝核首阶段由原种子的 '
            f'{original["first_trials"]:.3f} 次变为多种子均值 {multiple["first_trials"]:.3f} 次。'
            '这说明不能把原单轨迹尖峰直接解释为该子型的稳定难点；其他子型仍有起伏。',
            f'2. 少试探与每次更贵同时存在。k0=6 时，EP/投影拒绝核首阶段次数为 '
            f'{ep["first_trials"]:.3f}/{projection["first_trials"]:.3f}，每次线性消元为 '
            f'{ep["first_linear_ns_per_trial"]/1e6:.3f}/{projection["first_linear_ns_per_trial"]/1e6:.3f} ms。'
            '所以不能只用支撑覆盖概率推断总时间。',
            f'3. 统一消元之后，枚举收益仍然存在。k0=6 的EP拒绝/枚举在线均值为 '
            f'{ep["online_ns"]/1e6:.3f}/{ep_all["online_ns"]/1e6:.3f} ms；投影为 '
            f'{projection["online_ns"]/1e6:.3f}/{projection_all["online_ns"]/1e6:.3f} ms。'
            '原右列较快并非完全由两个消元程序写法不同造成。',
            '4. 整球收益不代表所有错误组成均有同等收益。含符号2的子组中，匹配枚举双方时间接近；'
            '当前同时区间未支持该子组中任何子型的EP或投影时间优势。a=2时低位为零，补全低位约束没有随机搜索可节省。',
            '5. k0=12 时，匹配双方的搜索计数和完整选集轨迹一致，时间均值也接近，边界对照通过。',
            '6. 本轮同时区间覆盖13个子型；原多半径分析覆盖52个半径/子型点。'
            '不能把两个分析中区间宽度或显著性标记的变化全部归因于增加搜索种子。', '']
    lines += ['## 共同消元下的主要结果', '',
        '以下均为整球组成加权均值；毫秒。', '',
        '| k0 | EP拒绝 | 投影拒绝 | EP枚举 | 投影枚举 | EP拒绝首阶段次数 | 投影拒绝首阶段次数 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for k0 in sorted({r['k0'] for r in table}):
        data = [lookup['common', m, 'all', 'sphere', k0] for m in MATCHED]
        lines.append(f'| {k0} | '+' | '.join(f'{r["online_ns"]/1e6:.3f}' for r in data)+
                     f' | {data[0]["first_trials"]:.3f} | {data[1]["first_trials"]:.3f} |')
    lines += ['', '## 解释拒绝核的代价', '',
        '下面的每次线性消元时间等于加权消元总时间/加权试探次数，包含建立受限增广矩阵和消元；不是逐次比值的平均。覆盖且奇异的选集在拒绝策略中被丢弃。', '',
        '| k0 | 方法 | 首阶段次数 | 奇异比例 | 覆盖且奇异次数 | 每次线性消元 ms |',
        '|---:|---|---:|---:|---:|---:|']
    for k0 in (0, 4, 5, 6, 7, 8, 12):
        if ('common', 'ep_reject', 'all', 'sphere', k0) not in lookup:
            continue
        for method in ('ep_reject', 'proj_binary_t'):
            row = lookup['common', method, 'all', 'sphere', k0]
            lines.append(f'| {k0} | {method} | {row["first_trials"]:.3f} | '
                         f'{100*row["first_singular_fraction"]:.2f}% | '
                         f'{row["first_covered_singular_trials"]:.3f} | '
                         f'{row["first_linear_ns_per_trial"]/1e6:.3f} |')
    lines += ['', '## 原种子与多种子：最低位试探次数', '',
        '两列处理完全相同的码和错误，仅搜索选集种子不同。该表用于判断原曲线尖峰是否对搜索随机性敏感，不增加码总体样本量。', '',
        '| k0 | EP原种子 | EP多种子 | 投影原种子 | 投影多种子 |',
        '|---:|---:|---:|---:|---:|']
    for k0 in sorted({r['k0'] for r in table}):
        values = [lookup['common', method, scope, 'sphere', k0]['first_trials']
                  for method in ('ep_reject', 'proj_binary_t') for scope in ('original', 'all')]
        lines.append(f'| {k0} | '+' | '.join(f'{value:.3f}' for value in values)+' |')
    lines += ['', '## 原实现与共同核：相同的原种子', '',
        '该比较包含实现差异，以及共同核为完成秩/一致性判断而保留的工作。共同核不是对优化拒绝算法最小运行时间的估计。', '',
        '| k0 | 方法 | legacy ms | common ms | 首阶段次数是否一致 |',
        '|---:|---|---:|---:|---|']
    for k0 in (0, 5, 6, 8, 12):
        if ('common', 'ep_reject', 'original', 'sphere', k0) not in lookup:
            continue
        for method in MATCHED:
            old = lookup['legacy', method, 'original', 'sphere', k0]
            new = lookup['common', method, 'original', 'sphere', k0]
            lines.append(f'| {k0} | {method} | {old["online_ns"]/1e6:.3f} | '
                         f'{new["online_ns"]/1e6:.3f} | '
                         f'{"是" if abs(old["first_trials"]-new["first_trials"]) < 1e-9 else "否"} |')
    lines += ['', '## 高位错误分组', '',
        'a 是符号 2 的个数。a>0 组按其真实球面条件概率加权；表中是完整在线译码时间。a=2 在 t=4 时低位为零，不代表一般低位搜索。', '',
        '| k0 | 组 | EP拒绝 ms | 投影拒绝 ms | EP枚举 ms | 投影枚举 ms |',
        '|---:|---|---:|---:|---:|---:|']
    for k0 in (0, 5, 6, 8, 12):
        if ('common', 'ep_reject', 'all', 'sphere', k0) not in lookup:
            continue
        for group in ('sphere', 'a_positive', 'a_two'):
            lines.append(f'| {k0} | {group} | '+' | '.join(
                f'{lookup["common", m, "all", group, k0]["online_ns"]/1e6:.3f}' for m in MATCHED)+' |')
    lines += ['', '## 配对时间比的同时区间', '',
        '比值为 EP 均值/投影均值。整条同时区间低于 1 才标记 EP 更低；高于 1 标记投影更低；跨 1 则尚未区分。没有通过区间判定不等于二者相同。', '',
        '| 策略 | 分组 | 支持EP较低的k0 | 支持投影较低的k0 |', '|---|---|---|---|']
    for label, text in (('common_reject_EP_vs_projection', '拒绝'),
                        ('common_enumerate_EP_vs_projection', '枚举')):
        for group in ('sphere', 'a_positive'):
            data = [r for r in ratios if r['comparison'] == label and r['group'] == group and r['metric'] == 'online_ns']
            lower = ', '.join(str(r['k0']) for r in data if r['result'] == 'numerator_lower') or '无'
            higher = ', '.join(str(r['k0']) for r in data if r['result'] == 'denominator_lower') or '无'
            lines.append(f'| {text} | {group} | {lower} | {higher} |')
    lines += ['', '## 图和数据用途', '',
        '* common_kernel_comparison：共同消元下的实际在线时间和首阶段试探次数，供主要方法比较。阴影是各均值的逐点95%区间。',
        '* seed_sensitivity：同一批码和错误，原种子与多种子首阶段次数的比较，用来定位原曲线起伏。此图不提供单点显著性判定。',
        '* paired_time_ratios：整球及 a>0 的配对时间比和同时95%带，用来判断哪些已测子型得到支持。',
        '* weighted_by_code.csv：每个母矩阵/子型的组成加权统计；summary.csv：母矩阵均值；paired_ratios.csv：配对区间；seed_variability.csv：同一错误不同搜索种子的均值、标准差、范围。',
        '* uniform_validation.csv：独立均匀球面附加输入的结果单列，未混入分层统计。', '',
        '* coverage_predictions.csv：枚举核的理论平均试探次数与观测均值。理论值由每种组成的支撑覆盖概率倒数加权得到，零伴随式和零外部半径的直接恢复记零次；不是耗时预测。', '',
        '## 解读范围', '',
        '本轮用于检验原结论对译码核实现和随机选集的敏感性；不能消除只有16个独立母矩阵、每个组成只有原有错误样本的限制。没有扫描其他n,K,t，没有新增DCC平均半径实验，没有加入共享预处理总成本。', '',
        '共同核即使采用拒绝政策，也会完成当前选集的消元与一致性检查；因此不能把它的耗时直接当作允许提前退出的最佳拒绝实现。', '',
        '复现：', '', '```powershell', 'python src/run_lee_fixed.py',
        'python src/analyze_lee_fixed.py results/lee_fixed_20261104', '```', '']
    (folder/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('At least two bootstrap replications are required')
    rows = [json.loads(line) for line in (args.folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    checked = audit(args.folder, rows)
    points = weighted(rows)
    table = summaries(points, args.bootstrap)
    ratios = comparisons(points, args.bootstrap)
    write_csv(args.folder/'weighted_by_code.csv', points)
    write_csv(args.folder/'summary.csv', table)
    write_csv(args.folder/'paired_ratios.csv', ratios)
    write_csv(args.folder/'seed_variability.csv', seed_variability(rows))
    write_csv(args.folder/'coverage_predictions.csv', coverage_predictions(table))
    uniform = []
    buckets = defaultdict(list)
    for row in rows:
        if row['population'] == 'uniform':
            buckets[row['kernel'], row['method'], row['k0']].append(row)
    for (kernel, method, k0), data in sorted(buckets.items()):
        uniform.append(dict(kernel=kernel, method=method, k0=k0, calls=len(data),
                            completed=sum(r['correct'] for r in data),
                            online_ms=statistics.mean(r['online_ns'] for r in data)/1e6,
                            first_trials=statistics.mean(r.get('first_trials', 0) for r in data)))
    write_csv(args.folder/'uniform_validation.csv', uniform)
    report(args.folder, table, ratios, checked)
    plots(args.folder, table, ratios)
    (args.folder/'analysis_manifest.json').write_text(json.dumps(dict(
        bootstrap=args.bootstrap,
        sources={name: hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                 for name in ('analyze_lee_fixed.py', 'analyze_lee_extended.py')},
        records_sha256=hashlib.sha256((args.folder/'instances.jsonl').read_bytes()).hexdigest(),
        figures=['common_kernel_comparison', 'seed_sensitivity', 'paired_time_ratios']),
        indent=2), encoding='utf-8')
    print(json.dumps(checked), flush=True)


if __name__ == '__main__':
    main()
