"""Reconstruct archived MMT success selections without running new searches."""
import argparse
from collections import Counter, defaultdict
import json
import random

import numpy as np

from analyze_lee_list_retuned import write_csv
from lee_extended import composition_probability
from lee_revision import eliminate
from lee_z4_dcc_comparison import Arithmetic, lee
from run_lee_list_confirmation import digest
from run_lee_list_retuned import ROOT, allocation, make_public


def binary_columns(matrix):
    return [sum((int(row[j]) & 1) << i for i, row in enumerate(matrix))
            for j in range(len(matrix[0]))]


def independent(columns, selected):
    pivots = {}
    for j in selected:
        value = columns[j]
        while value:
            bit = value.bit_length()-1
            if bit not in pivots:
                pivots[bit] = value
                break
            value ^= pivots[bit]
        if not value:
            return False
    return True


def effective_weights(params, radius, capacity, trial):
    if trial % 4 == 3:
        return 0, 0
    if 2*(params['weight']+params['weight2']) > radius:
        return 0, 0
    return min(params['weight'], capacity), min(params['weight2'], capacity)


def reconstruct(matrix, unit_rank, radius, seed, params, trials, modulus):
    """Only rank tests, RNG calls and the public weight schedule are replayed."""
    n = len(matrix[0])
    window = min(params['window'], unit_rank)
    columns = binary_columns(matrix)
    rng = random.Random(seed)
    counts = Counter()
    final = None
    for trial in range(trials):
        counts['trials'] += 1
        selected = rng.sample(range(n), unit_rank-window)
        if not independent(columns, selected):
            if trial == trials-1:
                raise AssertionError('Successful search ends with a singular selection')
            counts['singular_trials'] += 1
            continue
        counts['full_rank_trials'] += 1
        excluded = set(selected)
        free = [j for j in range(n) if j not in excluded]
        rng.shuffle(free)
        weights = effective_weights(params, radius, (1 if modulus == 2 else 2)*(len(free)//2), trial)
        counts['nonzero_list_trials' if sum(weights) else 'zero_list_trials'] += 1
        final = dict(selected=selected, free=free, weights=weights)
    if trials and (final is None or counts['full_rank_trials'] == 0):
        raise AssertionError('Successful search has no full-rank final selection')
    return final, counts


def phase_record(row, phase, matrix, unit_rank, radius, seed, word, modulus):
    counters, params = row['counters'], row['parameters']
    trials = counters.get(phase+'_trials', 0)
    if not trials:
        if any(word):
            raise AssertionError('Direct search returned a nonzero vector')
        return dict(phase=phase, mode='direct', terminal_free_weight=0, trials=0,
                    singular_trials=0, full_rank_trials=0,
                    nonzero_list_trials=0, zero_list_trials=0)
    final, counts = reconstruct(matrix, unit_rank, radius, seed, params, trials, modulus)
    for name in ('trials','singular_trials','full_rank_trials','nonzero_list_trials','zero_list_trials'):
        if counts[name] != counters.get(phase+'_'+name, 0):
            raise AssertionError('Reconstructed count differs from archive: '+phase+'_'+name)
    free_word = [word[j] for j in final['free']]
    free_weight = sum(free_word) if modulus == 2 else lee(free_word)
    if sum(final['weights']):
        mode = 'positive_free' if free_weight else 'nonzero_base_zero_free'
    else:
        if free_weight:
            raise AssertionError('Zero-base success has nonzero free part')
        mode = 'zero_base'
    return dict(phase=phase, mode=mode, terminal_free_weight=free_weight, **{
        name:counts[name] for name in ('trials','singular_trials','full_rank_trials',
                                      'nonzero_list_trials','zero_list_trials')})


def export_tables(summaries, tuning, output):
    lookup = {(r['target'],r['method']):r for r in summaries
              if r['variant']=='primary' and r['phase'] in ('first','native')}
    targets = sorted({r['target'] for r in summaries},
                     key=lambda ident:tuple(map(int,ident.split(':')[:4])))
    lines = [r'\begin{table}[t]',
        r'\caption{Successful MMT selections under positive-base parameter selection.',
        r'Entries are exact-sphere weighted percentages for the CCP low-digit',
        r'search and the native ring search. P: nonzero base weights and a',
        r'positive-weight accepted free part; C: nonzero base weights but a',
        r'zero accepted free part through cancellation; Z: zero base weights.',
        r'Direct zero-digit recovery accounts for the remaining fraction',
        r'(below $0.02\%$ for CCP). All percentages are rounded.}',
        r'\label{tab:leemmtsuccessmodes}',
        r'\centering\small\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{rrrrrrrrr}', r'\toprule',
        r'& & & \multicolumn{3}{c}{CCP low digit} & \multicolumn{3}{c}{Native}\\',
        r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
        r'$n$ & $t$ & $k_0$ & P & C & Z & P & C & Z\\', r'\midrule']
    for ident in targets:
        n,_,t,k0,_ = ident.split(':')
        values = [lookup[ident,method][name+'_fraction']*100
                  for method in ('ccp_mmt','lee_mmt')
                  for name in ('positive_free','nonzero_base_zero_free','zero_base')]
        lines.append(' & '.join([n,t,k0]+[f'{v:.1f}' for v in values])+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (output/'mmt_success_modes.tex').write_text('\n'.join(lines),encoding='utf-8')
    lines = [r'\begin{table}[t]',
        r'\caption{Frozen MMT parameters for Table~\ref{tab:leemmtstability}.',
        r'Each tuple gives the elimination window, the two base-list weights,',
        r'and the partial collision split. Base-list weights',
        r'are Hamming weights for CCP and Lee weights for native decoding.',
        r'The public zero-weight schedule and residual-radius adjustments',
        r'apply under both policies. Here $K=3n/16$.}',
        r'\label{tab:leemmtparameters}',r'\centering\small\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{rrrllll}',r'\toprule',
        r'& & & \multicolumn{2}{c}{Positive-base policy} & \multicolumn{2}{c}{Full family}\\',
        r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
        r'$n$ & $t$ & $k_0$ & CCP & Native & CCP & Native\\',r'\midrule']
    for ident in targets:
        n,_,t,k0,_ = ident.split(':')
        tuples = []
        for policy in ('primary','family'):
            for method in ('ccp_mmt','lee_mmt'):
                p = tuning[ident+':'+method][policy]
                tuples.append('$('+','.join(str(p[k]) for k in ('window','weight','weight2','split'))+')$')
        lines.append(' & '.join([n,t,k0]+tuples)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (output/'mmt_parameters.tex').write_text('\n'.join(lines),encoding='utf-8')


def analyze(folder):
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    for name, expected in config['source_hashes'].items():
        if digest(ROOT/'src'/name) != expected:
            raise AssertionError('Frozen source changed: '+name)
    rows = [json.loads(s) for s in (folder/'instances.jsonl').read_text(encoding='utf-8').splitlines()]
    mothers = {(m['n'],m['mother']):m for m in json.loads((folder/'mothers.json').read_text(encoding='utf-8'))
               if m['role']=='test'}
    publics, reconstructed, groups = {}, [], defaultdict(list)
    successes = {}
    for index, row in enumerate(rows):
        if row['status'] != 'complete' or not row['correct']:
            raise AssertionError('Mode attribution requires completed correct calls')
        key = row['n'], row['mother'], row['k0']
        if key not in publics:
            publics[key] = make_public(mothers[row['n'],row['mother']],row['k0'])[0]
        public = publics[key]
        if row['method']=='lee_mmt':
            stages = [('native',public['h'],public['units'],row['t'],row['search_seed'],row['error'],4)]
        else:
            low = [x % 2 for x in row['error']]
            stages = [('first',public['h0'],len(public['h0']),row['t'],row['search_seed'],low,2)]
            if row['counters'].get('recovery_trials',0):
                known = [j for j,x in enumerate(low) if x]
                reduced = eliminate(public['h1'], [0]*len(public['h1']), known, Arithmetic(1,False))
                if reduced is None:
                    raise AssertionError('Known columns dependent')
                outside = [j for j in range(row['n']) if j not in set(known)]
                h = [[r[j] for j in outside] for r in reduced[len(known):]]
                word = [row['error'][j]//2 for j in outside]
                stages.append(('recovery',h,len(h),row['a'],row['search_seed']+1,word,2))
            else:
                stages.append(('recovery',None,0,row['a'],row['search_seed']+1,[],2))
        for phase,matrix,rank,radius,seed,word,modulus in stages:
            # Variant repetitions with identical parameters have identical selections.
            cache_key = (row['target'],row['mother'],row['a'],row['case'],row['seed_index'],
                         row['method'],tuple(sorted(row['parameters'].items())),phase)
            if cache_key not in successes:
                successes[cache_key] = phase_record(row,phase,matrix,rank,radius,seed,word,modulus)
            value = successes[cache_key]
            for name in ('trials','singular_trials','full_rank_trials','nonzero_list_trials','zero_list_trials'):
                if value[name] != row['counters'].get(phase+'_'+name,0):
                    raise AssertionError('Cached reconstruction differs from archived counters')
            record = dict(target=row['target'],variant=row['variant'],method=row['method'],
                mother=row['mother'],a=row['a'],case=row['case'],seed_index=row['seed_index'],**value)
            reconstructed.append(record)
            groups[row['target'],row['variant'],row['method'],phase].append(record)
        if (index+1) % 2000 == 0:
            print('reconstructed',index+1,'/',len(rows),flush=True)
    summaries=[]
    phases = ('direct','positive_free','nonzero_base_zero_free','zero_base')
    for (ident,variant,method,phase),block in sorted(groups.items()):
        n,_,t,_,_ = ident.split(':')
        value = dict(target=ident,variant=variant,method=method,phase=phase)
        value.update({mode+'_fraction':0. for mode in phases})
        value.update(dict(trials=0.,nonzero_list_trials=0.,zero_list_trials=0.))
        for r in block:
            e,s=allocation(r['a'])
            weight=float(composition_probability(int(n),int(t),r['a']))/(config['test_mothers']*e*s)
            value[r['mode']+'_fraction'] += weight
            for metric in ('trials','nonzero_list_trials','zero_list_trials'):
                value[metric] += weight*r[metric]
        if not np.isclose(sum(value[mode+'_fraction'] for mode in phases),1):
            raise AssertionError('Success-mode probabilities do not sum to one')
        summaries.append(value)
    output=folder/'analysis'
    output.mkdir(exist_ok=True)
    write_csv(output/'success_modes_instances.csv',reconstructed)
    write_csv(output/'success_modes_summary.csv',summaries)
    export_tables(summaries,json.loads((folder/'tuning.json').read_text(encoding='utf-8')),output)
    (output/'success_modes_verification.json').write_text(json.dumps(dict(
        archived_calls=len(rows),reconstructed_stages=len(reconstructed),
        mode_definitions=dict(positive_free='Nonzero base weights and accepted free vector of positive metric weight.',
            nonzero_base_zero_free='Nonzero base weights but accepted free vector is zero after representation cancellation.',
            zero_base='Zero base weights on successful selection.',direct='No randomized search.'),
        weighting='Exact Lee-sphere probabilities, equal matrix and within-composition error/seed weights.',
        scope='Descriptive attribution for archived calls; no counterfactual cost reduction inferred.',
        source_hashes_match=True,rank_and_schedule_counts_match=True),indent=2),encoding='utf-8')
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',nargs='?',default='results/lee_mmt_stability_20261107')
    args=parser.parse_args()
    result=analyze(ROOT/args.folder)
    print(json.dumps([r for r in result if r['variant']=='primary'],indent=2))


if __name__=='__main__':
    main()
