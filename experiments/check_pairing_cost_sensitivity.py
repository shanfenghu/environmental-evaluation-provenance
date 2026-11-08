"""Fixed-state native timing probe and retained-bank influence diagnostics.

The probe uses four fixed states, ten paired repetitions and two warm-ups. Existing scientific outcomes are read-only. This is neither
a fresh assignment study nor a calibrated finite-bank precision comparison.
"""
import gc
import json
import os
from pathlib import Path
import time

# Fix native numerical parallelism before importing numerical libraries.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

from run_pairing_precision import ROOT, score_blocks
from source_assessment.data import Data
from source_assessment.io import sha
from source_assessment.models import fit_native, fit_median



def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite(x):
    return float(x) if np.isfinite(x) else None


def diagnostic(case):
    """Keep full estimates, all deletions and both first-plan strata visible."""
    banks = case['banks']
    scores = np.array([b['averaged'][5] for b in banks])
    order = np.array([b['order_variance'][5] for b in banks])
    fit = np.array([b['fit_seconds'] for b in banks])
    qc, qa = [np.array([b['query_seconds'][p] for b in banks])
              for p in ['cyclic', 'averaged']]
    vb, vo = scores.var(ddof=1), order.mean()
    a = vo / vb if vb > 0 else np.nan
    fstar = (qa.mean() - qc.mean()) / a - qc.mean() if a > 0 else np.inf
    result = {'workload': case['workload'], 'task': case['task'],
              'break_even_total_time_ratio': finite(1 + a),
              'mean_fit_seconds': float(fit.mean()),
              'mean_query_seconds': {'cyclic': float(qc.mean()), 'averaged': float(qa.mean())},
              'fit_seconds_at_break_even': finite(fstar),
              'fit_multiplier_at_break_even': finite(fstar / fit.mean()),
              'zero_fit_efficiency': finite((1 + a) * qc.mean() / qa.mean()),
              'maximum_centred_square_share': finite(((scores-scores.mean())**2).max() /
                                                     ((scores-scores.mean())**2).sum()),
              'delete_one_bank': [], 'first_plan_strata': {}}
    for excluded in range(len(banks)):
        keep = np.arange(len(banks)) != excluded
        vbd, vod = scores[keep].var(ddof=1), order[keep].mean()
        ratio = (fit[keep] + qa[keep]).mean() / (fit[keep] + qc[keep]).mean()
        result['delete_one_bank'].append({'excluded_bank': excluded,
            'bank_variance': float(vbd), 'mean_order_variance': float(vod),
            'time_ratio': float(ratio),
            'efficiency': finite((1 + vod/vbd) / ratio) if vbd > 0 else None})
    for first in ['cyclic', 'averaged']:
        states = [s for b in banks for s in b['states'] if s['first_plan'] == first]
        ratios = [s['query_seconds']['averaged']/s['query_seconds']['cyclic'] for s in states]
        result['first_plan_strata'][first] = {'states': len(states),
            'query_ratio': sum(s['query_seconds']['averaged'] for s in states) /
                           sum(s['query_seconds']['cyclic'] for s in states),
            'paired_ratio_range': [min(ratios), max(ratios)]}
    return result


def summarise(repeats):
    """Descriptive repeated-query times; the fitted state is the common unit."""
    values = repeats[2:]
    result = {'analysed_pairs': len(values), 'first_plan_strata': {}, 'cpu_wall_ratio': {}}
    ratios = [x['averaged']['wall_seconds']/x['cyclic']['wall_seconds'] for x in values]
    result['query_ratio'] = sum(x['averaged']['wall_seconds'] for x in values) / sum(
        x['cyclic']['wall_seconds'] for x in values)
    result['paired_ratio_median'] = float(np.median(ratios))
    result['paired_ratio_range'] = [min(ratios), max(ratios)]
    for plan in ['cyclic', 'averaged']:
        result['cpu_wall_ratio'][plan] = sum(x[plan]['cpu_seconds'] for x in values) / sum(
            x[plan]['wall_seconds'] for x in values)
        selected = [x for x in values if x['first_plan'] == plan]
        result['first_plan_strata'][plan] = {'pairs': len(selected),
            'query_ratio': sum(x['averaged']['wall_seconds'] for x in selected) / sum(
                x['cyclic']['wall_seconds'] for x in selected)}
    return result


def main():
    assert all(os.environ.get(k) == '1' for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'])
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', default='records/pairing')
    parser.add_argument('--data', default='data/prepared')
    parser.add_argument('--output', required=True)
    parser.add_argument('--recorded', help='Reconstruct timing summaries from an existing probe; perform no fits')
    args=parser.parse_args()
    REFERENCE=Path(args.records); OUTPUT=Path(args.output)
    run = json.loads((REFERENCE/'run.json').read_text())
    assert run['complete']
    cases = []
    for ref in run['cases']:
        assert sha(REFERENCE/ref['record']) == ref['sha256']
        cases.append(json.loads((REFERENCE/ref['record']).read_text()))
    OUTPUT.mkdir(parents=True,exist_ok=False)
    source_paths = [Path(__file__), ROOT/'experiments/run_pairing_precision.py',
                    ROOT/'requirements.lock', *sorted((ROOT/'src/source_assessment').glob('*.py'))]
    result = {'complete': False,
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        'reference_run_sha256': sha(REFERENCE/'run.json'),
        'scope': 'Retained-bank sensitivity plus four fixed-state repeated resident query probes; no new assignment or environmental replication',
        'retained_diagnostics': [diagnostic(c) for c in cases], 'native_probe': []}
    save(OUTPUT/'run.json', result)
    if args.recorded:
        recorded=json.loads(Path(args.recorded).read_text())
        assert recorded['complete']
        for probe in recorded['native_probe']:
            probe=dict(probe)
            probe['summary']=summarise(probe['repeats'])
            result['native_probe'].append(probe)
        result['complete']=True
        result['execution']='Reconstructed recorded timings; no fresh fits or timings'
        save(OUTPUT/'run.json', result)
        return
    recipes = run['recipes']
    for name in run['config']['workloads']:
        matching = [c for c in cases if c['workload'] == name]
        data = Data(name, args.data, {name: {'external': matching[0]['external']}})
        for ci, case in enumerate(matching):
            assert data.input_hash == case['input_sha256']
            old_path = Path(run['config']['reference_records'])/case['reference_case']
            assert sha(old_path) == case['reference_case_sha256']
            old = json.loads(old_path.read_text())
            bank = case['banks'][0]
            used = set(case['development']) - set(bank['blocks'][0])
            assert not used & set(case['external']) and set(case['protected']) <= used
            started = time.perf_counter()
            models = [fit_native(used, data.x, data.y, recipes[k]) for k in ['ridge', 'boosting']]
            medians = fit_median(used, data.meta, data.raw, data.month)
            fit_seconds = time.perf_counter() - started
            assert sha(REFERENCE/bank['loss_record']) == bank['loss_record_sha256']
            with np.load(REFERENCE/bank['loss_record'], allow_pickle=False) as archive:
                expected = dict(zip(bank['scored'], archive[bank['states'][0]['loss_key']]))
            probe = {'workload': name, 'task': case['task'], 'bank': 0, 'state': 0,
                     'fit_seconds': fit_seconds, 'repeats': [],
                     'reference_loss_sha256': bank['loss_record_sha256']}
            for repeat in range(10):
                first = 'cyclic' if (ci + repeat) % 2 == 0 else 'averaged'
                plans = [first, 'averaged' if first == 'cyclic' else 'cyclic']
                gc.collect()
                pair = {'repeat': repeat, 'warmup': repeat < 2, 'first_plan': first}
                for plan in plans:
                    selected = [0, 4] if plan == 'cyclic' else list(range(5))
                    wall, cpu = time.perf_counter(), time.process_time()
                    values, losses = score_blocks(data, models, medians, bank['blocks'], case['cells'], selected)
                    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                    discrepancy = max(float(np.max(abs(losses[i]-expected[i]))) for i in losses)
                    assert discrepancy < 1e-10
                    pair[plan] = {'wall_seconds': wall, 'cpu_seconds': cpu,
                        'max_reference_loss_discrepancy': discrepancy,
                        'prediction_rows': sum(len(data.w[i]) for row in selected for i in bank['blocks'][row])}
                    del values, losses
                probe['repeats'].append(pair)
            probe['summary'] = summarise(probe['repeats'])
            result['native_probe'].append(probe)
            save(OUTPUT/'run.json', result)
            print(name, case['task'], json.dumps(probe['summary']), flush=True)
            del models, medians
            gc.collect()
        del data
        gc.collect()
    result['complete'] = True
    save(OUTPUT/'run.json', result)
    print('COMPLETE four fixed states; 32 analysed paired repeats plus eight warm-ups', flush=True)


if __name__ == '__main__':
    main()
