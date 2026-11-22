"""Connect exact coverage predictions, measured searches, and online Lee cost."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from analyze_lee_extended import write_csv
from analyze_lee_fixed import METRICS, weighted
from lee_extended import composition_probability

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('ep_all', 'proj_binary_all')
SEEDS = 4
COLORS = {'ep_all': '#167397', 'proj_binary_all': '#c24c3b'}
LABELS = {'ep_all': 'EP', 'proj_binary_all': 'Ordinary projection'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_searches(n, k, t, k0, method, group='sphere'):
    """Exact selection expectations under uniqueness and singular enumeration."""
    if method not in METHODS:
        raise ValueError('Unknown method')
    initial_rows = n-k0 if method == 'ep_all' else n-k
    weights = [(a, float(composition_probability(n, t, a))) for a in range(t//2+1)
               if group == 'sphere' or (group == 'a_positive' and a > 0)]
    mass = sum(probability for _, probability in weights)
    if mass <= 0:
        raise ValueError('Empty error group')
    first, recovery = 0, 0
    for a, probability in weights:
        s = t-2*a
        if not probability:
            continue
        if s:
            if s > initial_rows:
                return math.inf, math.inf
            first += probability/mass*math.comb(n, s)/math.comb(initial_rows, s)
        if a:
            if a > n-k-s:
                return math.inf, math.inf
            recovery += probability/mass*math.comb(n-s, a)/math.comb(n-k-s, a)
    return first, recovery


def select_panel(rows):
    selected = [r for r in rows if r['kernel'] == 'common' and r['method'] in METHODS
                and r['population'] != 'uniform' and r['seed_index'] < SEEDS]
    pairs = defaultdict(dict)
    counts = Counter()
    for row in selected:
        if (row['n'], row['K'], row['t']) != (64, 12, 4):
            raise AssertionError('Unexpected parameters')
        if row['status'] != 'complete' or not row['correct']:
            raise AssertionError('Incomplete or incorrect measurement')
        if row['d_binary'] <= 16:
            raise AssertionError('Inherited distance condition failed')
        key = row['code'], row['population'], row['case'], row['seed_index']
        if row['method'] in pairs[key]:
            raise AssertionError('Duplicate method')
        pairs[key][row['method']] = row
        counts[row['code'], row['a'], row['method']] += 1
    if len({r['cluster'] for r in selected}) != 16 or len({r['code'] for r in selected}) != 208:
        raise AssertionError('Expected the complete original mother/subtype panel')
    if set(counts.values()) != {SEEDS}:
        raise AssertionError('Every code/composition must use exactly four seeds')
    for pair in pairs.values():
        if set(pair) != set(METHODS):
            raise AssertionError('Unpaired methods')
        ep, projection = (pair[m] for m in METHODS)
        for field in ('error', 'syndrome', 'trial_seed'):
            if ep[field] != projection[field]:
                raise AssertionError('Mismatched input/seed')
        for field in ep:
            if field.startswith('second_') and ep[field] != projection.get(field, 0):
                raise AssertionError('Recovery counters must match')
        if ep['k0'] == 12:
            for field in ('trace', 'first_trials', 'first_candidates', 'arithmetic_proxy'):
                if ep.get(field, 0) != projection.get(field, 0):
                    raise AssertionError('Free boundary mismatch')
    projection_counts = defaultdict(set)
    for row in selected:
        if row['method'] == 'proj_binary_all':
            key = row['cluster'], row['population'], row['case'], row['seed_index']
            projection_counts[key].add(row.get('first_trials', 0))
    if any(len(values) != 1 for values in projection_counts.values()):
        raise AssertionError('Projection counts differ across subtypes; draw separate intervals')
    return selected


def validate_stopping(rows):
    """With unique solutions, an all-solutions trial succeeds iff it covers."""
    counts = Counter()
    for row in rows:
        low = {j for j, value in enumerate(row['error']) if value % 2}
        rest = [j for j in range(len(row['error'])) if j not in low]
        high = {i for i, j in enumerate(rest) if row['error'][j] == 2}
        for phase, support, shortcut in (('first', low, not low), ('recovery', high, not high)):
            events = [e for e in row['trace'] if e['phase'] == phase]
            if shortcut:
                if events:
                    raise AssertionError('A zero-weight stage should not sample')
                counts[phase+'_shortcuts'] += 1
                continue
            if not events:
                raise AssertionError('Nonzero error needs a selection')
            for index, event in enumerate(events):
                covered = support.issubset(event['chosen'])
                if event['covers_true_support'] != covered:
                    raise AssertionError('Stored coverage differs')
                if covered != (index == len(events)-1):
                    raise AssertionError('Selection stopping differs from first support coverage')
                if covered and not event['consistent']:
                    raise AssertionError('Covered selection cannot be inconsistent')
                counts[phase+'_selections'] += 1
                if covered and event['nullity']:
                    counts[phase+'_successful_singular_selections'] += 1
            counts[phase+'_nonzero_searches'] += 1
    return dict(counts)


def summarize(points, repetitions):
    lookup = {(r['cluster'], r['k0'], r['group'], r['method']): r for r in points}
    clusters = sorted({r['cluster'] for r in points})
    ks = list(range(13))
    n = len(clusters)
    rng = np.random.default_rng(2026110409)
    counts = rng.multinomial(n, np.full(n, 1/n), size=repetitions)
    output, ratios, calibration = [], [], []
    for group in ('sphere', 'a_positive'):
        arrays = {}
        for method in METHODS:
            for metric in METRICS:
                arrays[method, metric] = np.array([[lookup[c, k0, group, method][metric]
                                                   for k0 in ks] for c in clusters])
            for index, k0 in enumerate(ks):
                item = dict(k0=k0, extra_constraints=12-k0, group=group, method=method,
                            mothers=n, search_seeds=SEEDS)
                for metric in METRICS:
                    item[metric] = float(arrays[method, metric][:, index].mean())
                first, recovery = expected_searches(64, 12, 4, k0, method, group)
                item.update(predicted_first_trials=first, predicted_recovery_trials=recovery)
                for metric in ('online_ns', 'first_trials', 'recovery_trials'):
                    values = arrays[method, metric][:, index]
                    draws = counts @ values/n
                    low, high = np.quantile(draws, [.025, .975])
                    item[metric+'_low95'], item[metric+'_high95'] = float(low), float(high)
                output.append(item)
            # A separate simultaneous prediction check for each method/group/stage.
            for metric, stage in (('first_trials', 'first'), ('recovery_trials', 'recovery')):
                values = arrays[method, metric]
                mean = values.mean(axis=0)
                predicted = np.array([expected_searches(64, 12, 4, k, method, group)[0 if stage == 'first' else 1]
                                      for k in ks])
                draws = counts @ values/n
                deviation = np.max(np.abs(np.log(draws)-np.log(mean)), axis=1)
                width = float(np.quantile(deviation, .95))
                for index, k0 in enumerate(ks):
                    low, high = mean[index]*np.exp(-width), mean[index]*np.exp(width)
                    calibration.append(dict(method=method, group=group, stage=stage, k0=k0,
                        prediction=float(predicted[index]), observed=float(mean[index]),
                        observed_over_prediction=float(mean[index]/predicted[index]),
                        simultaneous_low95=float(low), simultaneous_high95=float(high),
                        prediction_inside=bool(low <= predicted[index] <= high)))
        for metric in ('online_ns', 'first_trials', 'first_ns', 'recovery_ns', 'arithmetic_proxy'):
            a, b = (arrays[m, metric] for m in METHODS)
            ratio = a.mean(axis=0)/b.mean(axis=0)
            draws = (counts @ a)/(counts @ b)
            width = float(np.quantile(np.max(np.abs(np.log(draws)-np.log(ratio)), axis=1), .95))
            low, high = np.quantile(draws, [.025, .975], axis=0)
            for index, k0 in enumerate(ks):
                slo, shi = ratio[index]*np.exp(-width), ratio[index]*np.exp(width)
                ratios.append(dict(k0=k0, extra_constraints=12-k0, group=group, metric=metric,
                    ratio=float(ratio[index]), low95=float(low[index]), high95=float(high[index]),
                    simultaneous_low95=float(slo), simultaneous_high95=float(shi),
                    result='EP_lower' if shi < 1 else 'projection_lower' if slo > 1 else 'unresolved'))
    return output, ratios, calibration


def plot_main(folder, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for method, marker in zip(METHODS, ('o', 's')):
        data = sorted((r for r in table if r['method'] == method and r['group'] == 'sphere'),
                      key=lambda r: r['extra_constraints'])
        x = [r['extra_constraints'] for r in data]
        color = COLORS[method]
        theory = [r['predicted_first_trials'] for r in data]
        axes[0].plot(x, theory, '--', color=color, linewidth=1.7)
        for ax, metric, scale in ((axes[0], 'first_trials', 1), (axes[1], 'online_ns', 1e6)):
            mean = np.array([r[metric]/scale for r in data])
            low = np.array([r[metric+'_low95']/scale for r in data])
            high = np.array([r[metric+'_high95']/scale for r in data])
            ax.errorbar(x, mean, yerr=np.array([mean-low, high-mean]), fmt=marker,
                        color=color, markersize=4.5, capsize=2, elinewidth=.7,
                        errorevery=len(x) if metric == 'first_trials' and method == 'proj_binary_all' else 1,
                        label=LABELS[method], zorder=3)
            if metric == 'online_ns' or method == 'proj_binary_all':
                ax.plot(x, mean, color=color, linewidth=1)
    axes[0].set_title('(a) Initial search: prediction and measurement')
    axes[1].set_title('(b) Complete online decoding cost')
    axes[0].set_ylabel('Mean first-stage selections')
    axes[1].set_ylabel('Mean online time (ms)')
    method_handles = [Line2D([0], [0], color=COLORS[m], marker=marker, linestyle='None', label=LABELS[m])
                      for m, marker in zip(METHODS, ('o', 's'))]
    axes[0].legend(handles=method_handles+[
        Line2D([0], [0], color='.35', linestyle='--', label='Exact coverage prediction')],
        loc='upper right', frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel(r'Additional constraints $K-k_0$')
        ax.set_xticks(range(0, 13, 2))
        ax.set_xlim(-.35, 12.35)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='.9', linewidth=.6)
    for extension in ('png', 'pdf'):
        fig.savefig(folder/f'search_to_cost.{extension}', dpi=220)
    plt.close(fig)


def stage_table(table):
    lookup = {(r['k0'], r['method'], r['group']): r for r in table}
    result = []
    for k0 in (0, 4, 8, 12):
        for method in METHODS:
            row = lookup[k0, method, 'sphere']
            high = lookup[k0, method, 'a_positive']
            result.append(dict(k0=k0, extra_constraints=12-k0, method=LABELS[method],
                predicted_first_trials=row['predicted_first_trials'],
                measured_first_trials=row['first_trials'],
                first_ms=row['first_ns']/1e6, recovery_ms=row['recovery_ns']/1e6,
                verify_ms=row['verify_ns']/1e6, online_ms=row['online_ns']/1e6,
                arithmetic_proxy_million=row['arithmetic_proxy']/1e6,
                high_group_online_ms=high['online_ns']/1e6,
                high_group_recovery_ms=high['recovery_ns']/1e6))
    return result


def write_report(folder, table, stages, ratios, calibration, check):
    lookup = {(r['k0'], r['method'], r['group']): r for r in table}
    ep, projection = (lookup[6, m, 'sphere'] for m in METHODS)
    lines = ['# 从覆盖预测到实测搜索，再到完整译码成本', '',
        '固定 R=Z4、n=64、K=12、t=4。本轮复用已经完成并核查的实测记录，新增逐轨迹停止条件验证、统一样本数的统计分析和论文展示文件，没有重新计时、生成新码或错误。', '',
        '## 主图比较什么', '',
        '![覆盖预测与实际成本](search_to_cost.png)', '',
        '* 蓝色始终表示EP，红色始终表示普通投影。双方首阶段半径均为t，采用同一个受限矩阵消元核、枚举一致奇异系统和条件二元恢复。普通投影不是DCC平均阈值实现。',
        '* 横轴是额外约束数 K-k0，从左到右约束增多；最左端k0=12是双方约束相同的边界，最右端k0=0的EP最低位被线性方程唯一确定。',
        '* 左图：虚线是精确覆盖模型预测的首阶段平均选集次数，点是实测均值，误差棒是按母矩阵重抽样得到的逐点95%区间。',
        '* 普通投影首阶段的选集次数在全部子型上逐输入相同，因此左图只在额外约束数为0处画一次公共误差棒，避免重复。',
        '* 右图：完整在线时间，包含构造、搜索、恢复和最终验证，排除预处理；点、线和误差棒使用相同方法颜色。',
        '* 全部13个子型统一使用前4个搜索种子，避免原4/8种子混合导致基准曲线出现人为台阶。额外种子仍保留在原诊断数据中。', '',
        '## 覆盖预测的推导', '',
        '令a为符号2的数量，s=4-2a为最低位的Hamming重量。精确Lee球面上的组成概率为', '',
        '$$w_a=\\frac{\\binom{64}{4-2a}\\binom{60+2a}{a}2^{4-2a}}{\\binom{128}{4}},\\quad a=0,1,2.$$', '',
        'EP和普通投影的选集大小分别为64-k0和52。给定s>0，选集覆盖真实最低位支撑的概率为', '',
        '$$P_{\\rm EP}(s)=\\frac{\\binom{64-k_0}{s}}{\\binom{64}{s}},\\qquad '
        'P_{\\rm Proj}(s)=\\frac{\\binom{52}{s}}{\\binom{64}{s}}.$$', '',
        '唯一性保证没有另一个半径内候选。枚举全部一致受限解时，一次选集找到真实低位当且仅当覆盖其支撑，故非零低位的平均选集次数为1/P。s=0时程序走零伴随式捷径，记零次，而不是一次。按w_a加权得到左图虚线。', '',
        '高位在已恢复支撑之外有a个非零位置，长度64-s，选集大小52-s。a>0时平均选集次数为binom(64-s,a)/binom(52-s,a)；a=0直接恢复，记零次。匹配双方高位系统、半径和种子相同。', '',
        '这些预测是当前全解核下的精确随机选集期望；不是墙钟时间公式，也没有把重量测试、消元和候选枚举成本当作相同常数。', '',
        '## 从理论次数到实际时间', '',
        f'例如k0=6：EP理论首阶段次数为{ep["predicted_first_trials"]:.3f}，实测为{ep["first_trials"]:.3f}；'
        f'普通投影理论为{projection["predicted_first_trials"]:.3f}，实测为{projection["first_trials"]:.3f}。', '',
        f'该点完整在线时间为EP {ep["online_ns"]/1e6:.3f} ms、普通投影 {projection["online_ns"]/1e6:.3f} ms。'
        '次数收益不会等比例转化成时间收益，因为EP每次消元的系统更大，完整译码还包括高位恢复和验证。', '',
        '## 阶段成本小表', '',
        '取等间隔子型k0=0,4,8,12，包括两个边界。代理以百万为单位；高位组表示至少含一个符号2的条件均值。', '',
        '| k0 | 方法 | 低位 ms | 恢复 ms | 完整在线 ms | 算术代理/百万 | 高位组完整在线 ms |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for row in stages:
        lines.append(f'| {row["k0"]} | {row["method"]} | {row["first_ms"]:.3f} | '
            f'{row["recovery_ms"]:.3f} | {row["online_ms"]:.3f} | '
            f'{row["arithmetic_proxy_million"]:.3f} | {row["high_group_online_ms"]:.3f} |')
    lines += ['', '阶段时间分别取重复运行中位数，完整时间还包括验证和外层开销，因此不要求表中低位时间加恢复时间严格等于完整在线时间。算术代理按原加/乘/逆权重计费，不是CPU指令数。', '',
        '## 统一样本后的区间判断', '',
        'paired_ratios.csv保存配对均值比及其同时95%区间；每个比较指标和错误分组分别覆盖全部13个子型。主图误差棒仅用于显示均值精度，不根据两条误差棒是否重叠判断优势。', '',
        '| 分组 | 支持EP在线更低的k0 | 支持普通投影在线更低的k0 |', '|---|---|---|']
    for group in ('sphere', 'a_positive'):
        data = [r for r in ratios if r['group'] == group and r['metric'] == 'online_ns']
        ep_points = ', '.join(str(r['k0']) for r in data if r['result'] == 'EP_lower') or '无'
        projection_points = ', '.join(str(r['k0']) for r in data if r['result'] == 'projection_lower') or '无'
        lines.append(f'| {group} | {ep_points} | {projection_points} |')
    lines += ['', '## 预测与轨迹核查', '',
        f'* 本轮选用{check["selected_records"]:,}条既有计数记录，对应{check["selected_timed_calls"]:,}次既有计时运行；'
        f'没有新增运行。这些记录包含{check["mothers"]}个独立母矩阵、208个相关子型码。',
        '* 每一条非零低位/高位搜索轨迹，都在第一次覆盖真实支撑的选集停止；此前选集均不覆盖。覆盖检查在译码结束后进行，未给译码器真实支撑。',
        f'* 共有{check["stopping"]["first_successful_singular_selections"]}次低位搜索及'
        f'{check["stopping"]["recovery_successful_singular_selections"]}次高位搜索在奇异选集上成功，验证枚举核确实利用了这些选集。'
        '这是分层样本的原始计数，不是球面加权概率。',
        '* k0=12的匹配计数和选集轨迹完全一致；高位恢复计数在每一对EP/投影输入上相同。',
        '* prediction_check.csv保存观测/理论次数之比及同时区间；观测期望与有限种子样本均值不会逐点相等。', '',
        f'* {len(calibration)}个方法/分组/阶段/子型预测值中，'
        f'{sum(r["prediction_inside"] for r in calibration)}个位于相应观测均值的同时区间内。'
        '同时区间按方法、分组和阶段分别构造，没有声称对全部统计族联合控制误差。', '',
        '## 如何用于论文', '',
        '主图把机制和代价连起来：补全约束允许更大的选集，提高支撑覆盖概率，减少最低位试探；完整时间同时受到每次消元规模以及恢复/验证成本影响。'
        '分组表保留含符号2的输入，说明整球收益是否也存在于需要高位外部位置恢复的输入。', '',
        '当前固定16个母矩阵和原错误样本，不能据此界定连续参数区间。最右端k0=0与最左端自由码均为边界对照，中间子型用于观察混合结构。', '',
        '## 文件', '',
        '* search_to_cost.png/.pdf：两面板主图。',
        '* stage_costs.csv/.tex：阶段成本与高位分组小表，可独立引用；没有修改论文tex。',
        '* panel_summary.csv：统一4种子后的全部13个子型统计，含理论次数。',
        '* weighted_by_code.csv：每个母矩阵的加权统计。',
        '* paired_ratios.csv：配对均值比、逐点区间和同时区间。',
        '* prediction_check.csv：覆盖预测与观测次数的核查。',
        '* validation.json、manifest.json：核查范围及数据/代码哈希。', '',
        '复现：`python src/present_lee_fixed.py`。', '']
    (folder/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    tex = [r'\begin{tabular}{rlrrrrr}', r'\toprule',
           r'$k_0$ & Method & Low (ms) & Recovery (ms) & Online (ms) & Proxy ($10^6$) & Online, $a>0$ (ms) \\',
           r'\midrule']
    for row in stages:
        tex.append(f'{row["k0"]} & {row["method"]} & {row["first_ms"]:.3f} & '
                   f'{row["recovery_ms"]:.3f} & {row["online_ms"]:.3f} & '
                   f'{row["arithmetic_proxy_million"]:.3f} & {row["high_group_online_ms"]:.3f} '+r'\\')
    tex += [r'\bottomrule', r'\end{tabular}', '']
    (folder/'stage_costs.tex').write_text('\n'.join(tex), encoding='utf-8')
    (folder/'caption.tex').write_text(
        'Fixed-parameter Lee decoding over $\\mathbb Z_4$ with $n=64$, $K=12$, and $t=4$. '
        'Both methods use the same binary all-solutions kernel and conditional high-digit recovery. '
        '(a) Exact support-coverage predictions (dashed) and measured first-stage selection counts (markers). '
        '(b) Complete online decoding time, excluding preprocessing. '
        'Each subtype uses the same four search-seed indices on the archived errors from 16 mother matrices '
        'conditioned on $d_H(D_1)>16$. Errors are weighted by their exact Lee-sphere composition probabilities. '
        'Error bars are pointwise 95\\% mother-cluster bootstrap intervals; the identical projection '
        'selection-count interval is drawn only at zero additional constraints.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/lee_fixed_20261104')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error('At least two bootstrap replications are required')
    source = args.source.resolve()
    output = args.output.resolve() if args.output else source/'presentation'
    if output == source:
        raise ValueError('Use a separate presentation subdirectory')
    audit = json.loads((source/'audit.json').read_text(encoding='utf-8'))
    manifest = json.loads((source/'analysis_manifest.json').read_text(encoding='utf-8'))
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if audit['status'] != 'passed' or digest(source/'instances.jsonl') != manifest['records_sha256']:
        raise AssertionError('Measured source has not passed unchanged audit')
    for name, expected in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Measured implementation changed: '+name)
    for name, expected in manifest['sources'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Analysis source changed: '+name)
    rows = [json.loads(line) for line in (source/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = select_panel(rows)
    stopping = validate_stopping(selected)
    points = [p for p in weighted(selected) if p['seed_scope'] == 'all' and p['group'] in ('sphere', 'a_positive')]
    table, ratios, calibration = summarize(points, args.bootstrap)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'weighted_by_code.csv', points)
    write_csv(output/'panel_summary.csv', table)
    write_csv(output/'paired_ratios.csv', ratios)
    write_csv(output/'prediction_check.csv', calibration)
    stages = stage_table(table)
    write_csv(output/'stage_costs.csv', stages)
    check = dict(status='passed', n=64, K=12, t=4, mothers=16, codes=208,
                 seeds_per_error=SEEDS, selected_records=len(selected),
                 selected_timed_calls=sum(len(r['timing_samples']) for r in selected),
                 new_decoding_calls=0, new_codes=0, new_errors=0, stopping=stopping)
    (output/'validation.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
    write_report(output, table, stages, ratios, calibration, check)
    plot_main(output, table)
    (output/'manifest.json').write_text(json.dumps(dict(
        source=str(source), bootstrap=args.bootstrap, seeds=SEEDS,
        source_hashes={name: digest(source/name) for name in
                       ('instances.jsonl', 'config.json', 'audit.json', 'analysis_manifest.json')},
        code_hashes={name: digest(ROOT/'src'/name) for name in
                     ('present_lee_fixed.py', 'analyze_lee_fixed.py', 'analyze_lee_extended.py')},
        output_hashes={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}),
        indent=2), encoding='utf-8')
    print(json.dumps(check), flush=True)


if __name__ == '__main__':
    main()
