"""Analyse the prespecified fresh-bank variance/cost comparison without fitting."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from source_assessment.io import sha


def measurements(average, order_variance, cyclic_time, averaged_time):
    """Moment/cost plug-ins for a common-law bank; undefined ratios stay explicit."""
    bank = np.var(average, axis=0, ddof=1)
    order = np.mean(order_variance, axis=0)
    total = bank + order
    time_ratio = float(np.mean(averaged_time) / np.mean(cyclic_time))
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = order / total
        efficiency = total / bank / time_ratio
    return bank, order, fraction, time_ratio, efficiency


def finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def finite_json(value):
    if isinstance(value, list):
        return [finite_json(x) for x in value]
    return value if np.isfinite(value) else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', default='records/pairing')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.records)
    run = json.loads((directory / 'run.json').read_text())
    assert run['complete']
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Preserve existing analysis')
    cfg = run['config']
    result = {'scope': run['scope'], 'run_sha256': sha(directory / 'run.json'),
              'columns': ['ridge', 'boosting', 'seasonal', 'ridge_minus_seasonal',
                          'boosting_minus_seasonal', 'boosting_minus_ridge'],
              'primary_coordinate': 'boosting_minus_ridge',
              'bootstrap_scope': 'Paired whole-bank marginal descriptive 95% ranges; no finite-sample/population/across-task coverage guarantee',
              'cases': []}
    for ci, ref in enumerate(run['cases']):
        assert sha(directory / ref['record']) == ref['sha256']
        case = json.loads((directory / ref['record']).read_text())
        banks = case['banks']
        assert len(banks) == cfg['banks']
        a = np.array([x['averaged'] for x in banks])
        c = np.array([x['cyclic'] for x in banks])
        v = np.array([x['order_variance'] for x in banks])
        tc, ta = [np.array([x['fit_plus_query_seconds'][plan] for x in banks])
                  for plan in ['cyclic', 'averaged']]
        vb, vo, fraction, time_ratio, efficiency = measurements(a, v, tc, ta)
        rng = np.random.default_rng(cfg['bootstrap_seed'] + ci)
        boot = [[], [], []]
        for ids in rng.integers(0, len(a), (cfg['bootstrap_replicates'], len(a))):
            _, _, f, t, e = measurements(a[ids], v[ids], tc[ids], ta[ids])
            boot[0].append(f); boot[1].append(t); boot[2].append(e)
        intervals = {}
        for name, values in zip(['order_fraction', 'time_ratio', 'efficiency'], boot):
            values = np.asarray(values)
            intervals[name] = {'low': finite_json(np.nanquantile(np.where(np.isfinite(values), values, np.nan), .025, axis=0).tolist()),
                               'high': finite_json(np.nanquantile(np.where(np.isfinite(values), values, np.nan), .975, axis=0).tolist()),
                               'nonfinite_replicates': np.sum(~np.isfinite(values), axis=0).tolist()}
        query = {plan: np.array([x['query_seconds'][plan] for x in banks])
                 for plan in ['cyclic', 'averaged']}
        item = {'workload': case['workload'], 'task': case['task'], 'banks': len(banks),
                'cyclic_mean': c.mean(axis=0).tolist(), 'averaged_mean': a.mean(axis=0).tolist(),
                'mean_paired_difference': (c-a).mean(axis=0).tolist(),
                'cyclic_observed_sd': c.std(axis=0, ddof=1).tolist(),
                'averaged_observed_sd': a.std(axis=0, ddof=1).tolist(),
                'cyclic_minimum': c.min(axis=0).tolist(), 'averaged_minimum': a.min(axis=0).tolist(),
                'minimum_over_all_orders': np.min([x['minimum_over_orders'] for x in banks], axis=0).tolist(),
                'maximum_over_all_orders': np.max([x['maximum_over_orders'] for x in banks], axis=0).tolist(),
                'bank_variance': vb.tolist(), 'mean_order_variance': vo.tolist(),
                'order_variance_fraction': finite_list(fraction),
                'fit_plus_query_time_ratio': time_ratio,
                'efficiency_ratio': finite_list(efficiency),
                'efficiency_status': ['finite' if np.isfinite(x) else
                                      ('infinite' if np.isinf(x) else 'undefined')
                                      for x in efficiency],
                'undefined_efficiency_coordinates': np.flatnonzero(~np.isfinite(efficiency)).tolist(),
                'mean_fit_seconds': float(np.mean([x['fit_seconds'] for x in banks])),
                'mean_query_seconds': {p: float(x.mean()) for p, x in query.items()},
                'mean_fit_plus_query_seconds': {'cyclic': float(tc.mean()), 'averaged': float(ta.mean())},
                'query_only_time_ratio': float(query['averaged'].mean()/query['cyclic'].mean()),
                'prediction_row_ratio': 2.5, 'intervals': intervals}
        result['cases'].append(item)
        print(item['workload'], item['task'], 'order fraction', fraction[5],
              'time ratio', time_ratio, 'efficiency', efficiency[5], flush=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
