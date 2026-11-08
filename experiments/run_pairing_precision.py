"""Fresh conditional banks comparing native cyclic and all-other score queries.

The configuration fixes the assignment seeds and timing boundary. Inputs and recipes
remain unchanged; source-indexed losses permit independent matrix reconstruction.
"""
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

# Fix native numerical parallelism before importing numerical libraries.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from source_assessment.data import Data
from source_assessment.io import sha
from source_assessment.models import fit_native, fit_median, predict
from source_assessment.pairing import paired_audit, averaged_audit
from source_assessment.queries import expand
from source_assessment.sampling import refreshed_folds




def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reference_cases(directory, references):
    """Read authenticated case contents; the manifest need only name/hash them."""
    cases = {}
    for reference in references:
        path = directory / reference['record']
        assert sha(path) == reference['sha256']
        case = json.loads(path.read_text())
        key = case['workload'], case['task']
        assert key not in cases
        cases[key] = case
    return cases


def score_blocks(data, models, medians, blocks, cells, selected):
    """Predict selected later blocks and return raw equal-cell/source/day losses.

    All three learners share the exact rows. The caller times this whole function,
    including concatenation and aggregation, for either query plan.
    """
    selected = sorted(selected)
    ids = sorted(i for row in selected for i in blocks[row])
    assert len(ids) == len(set(ids))
    xx = np.concatenate([data.z[i] for i in ids])
    predictions = [predict(model, xx, data.hydro) for model in models]
    losses, offset = {}, 0
    for i in ids:
        stop = offset + len(data.w[i])
        seasonal = np.array([medians[data.meta[i]['cell'], int(m)]
                             for m in data.future[i]])
        losses[i] = np.array([np.mean(abs(p[offset:stop] - data.w[i]))
                              for p in predictions] +
                             [np.mean(abs(seasonal - data.w[i]))])
        offset = stop
    values = {}
    for row in selected:
        values[row] = np.mean([
            np.mean([losses[i] for i in blocks[row] if i in group], axis=0)
            for group in cells.values()], axis=0)
    assert np.isfinite(np.array(list(losses.values()))).all()
    return values, losses


def main():
    assert all(os.environ.get(k) == '1'
               for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'])
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pairing-v1.json')
    parser.add_argument('--data', default='data/prepared')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    original_cfg = Path(cfg['reference_config'])
    recipes = json.loads(original_cfg.read_text())['learners']
    source = Path(cfg['reference_records'])
    original = json.loads((source / 'run.json').read_text())
    cases = reference_cases(source, original['cases'])
    old_seeds = {c['seed'] for value in cases.values() for c in value['cycles']}
    new_seeds = [cfg['seed'] + 100000 * wi + 10000 * ti + bank
                 for wi in range(2) for ti in range(2) for bank in range(cfg['banks'])]
    assert len(set(new_seeds)) == len(new_seeds) and not set(new_seeds) & old_seeds
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    source_paths = [Path(__file__), *sorted((ROOT/'src/source_assessment').glob('*.py')), ROOT/'requirements.lock']
    run = {'complete': False,
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        'reference_run_sha256': sha(source / 'run.json'),
        'config': cfg, 'recipes': recipes, 'cases': [],
        'scope': 'Fresh conditional assignments of inspected records; paired resident native query costs; not new environmental data or a cold-cache benchmark'}
    save(out / 'run.json', run)
    for wi, name in enumerate(cfg['workloads']):
        old_disperse = cases[name, 'dispersed']
        data = Data(name, args.data,
                    {name: {'external': old_disperse['external']}})
        assert data.input_hash == old_disperse['input_sha256']
        # Array content is exact; metadata may omit storage-only provenance.
        for ti, task in enumerate(cfg['tasks']):
            old = cases[name, task]
            dev, external, protected = map(set, [old['development'], old['external'], old['protected']])
            cells, quotas = old['cells'], old['quotas']
            record = {'workload': name, 'task': task, 'input_sha256': data.input_hash,
                      'reference_case': f'case-{wi}-{ti}.json',
                      'reference_case_sha256': sha(source / f'case-{wi}-{ti}.json'),
                      'development': sorted(dev), 'external': sorted(external),
                      'protected': sorted(protected), 'cells': cells, 'quotas': quotas,
                      'banks': []}
            for bank in range(cfg['banks']):
                seed = cfg['seed'] + wi * 100000 + ti * 10000 + bank
                blocks = list(refreshed_folds(cells, quotas, cfg['folds'], seed))
                scored = sorted(set().union(*map(set, blocks)))
                assert len(scored) == sum(map(len, blocks))
                matrix = np.empty((cfg['folds'], cfg['folds'], 3))
                state_records, source_losses = [], {}
                for j in range(cfg['folds']):
                    used = dev - set(blocks[j])
                    assert protected <= used and not used & external
                    started = time.perf_counter()
                    models = [fit_native(used, data.x, data.y, recipes[k])
                              for k in ['ridge', 'boosting']]
                    medians = fit_median(used, data.meta, data.raw, data.month)
                    fit_seconds = time.perf_counter() - started
                    first = 'cyclic' if (wi + ti + bank + j) % 2 == 0 else 'averaged'
                    plans = [first, 'averaged' if first == 'cyclic' else 'cyclic']
                    values, losses, timings, counts = {}, {}, {}, {}
                    for plan in plans:
                        selected = [j, (j - 1) % cfg['folds']] if plan == 'cyclic' else list(range(cfg['folds']))
                        started = time.perf_counter()
                        values[plan], losses[plan] = score_blocks(
                            data, models, medians, blocks, cells, selected)
                        timings[plan] = time.perf_counter() - started
                        counts[plan] = sum(len(data.w[i]) for row in selected for i in blocks[row])
                    discrepancy = max(float(np.max(abs(losses['cyclic'][i] - losses['averaged'][i])))
                                      for i in losses['cyclic'])
                    assert discrepancy < 1e-10
                    for i in range(cfg['folds']):
                        matrix[i, j] = values['averaged'][i]
                    key = f'{bank}_{j}'
                    source_losses[key] = np.array([losses['averaged'][i] for i in scored])
                    state_records.append({'excluded': blocks[j], 'fitted_count': len(used),
                        'loss_key': key, 'fit_seconds': fit_seconds, 'query_seconds': timings,
                        'prediction_rows': counts, 'first_plan': first,
                        'max_common_loss_discrepancy': discrepancy})
                    del models, medians, values, losses
                    gc.collect()
                    print(name, task, 'bank', bank + 1, '/', cfg['banks'], 'state', j + 1, '/5', flush=True)
                g = expand(matrix)
                orders = np.array([paired_audit(g, (0,) + p)
                                   for p in itertools.permutations(range(1, cfg['folds']))])
                average = averaged_audit(g)
                assert np.max(abs(orders.mean(axis=0) - average)) < 1e-10
                fit = sum(x['fit_seconds'] for x in state_records)
                query = {plan: sum(x['query_seconds'][plan] for x in state_records)
                         for plan in ['cyclic', 'averaged']}
                rows = {plan: sum(x['prediction_rows'][plan] for x in state_records)
                        for plan in query}
                assert 2 * rows['averaged'] == 5 * rows['cyclic']
                loss_path = out / f'losses-{wi}-{ti}-{bank}.npz'
                np.savez_compressed(loss_path, **source_losses)
                record['banks'].append({'bank': bank, 'seed': seed, 'blocks': blocks,
                    'scored': scored, 'matrix': g.tolist(), 'states': state_records,
                    'cyclic': paired_audit(g).tolist(), 'averaged': average.tolist(),
                    'order_variance': orders.var(axis=0).tolist(),
                    'minimum_over_orders': orders.min(axis=0).tolist(),
                    'maximum_over_orders': orders.max(axis=0).tolist(),
                    'fit_seconds': fit, 'query_seconds': query,
                    'fit_plus_query_seconds': {plan: fit + query[plan] for plan in query},
                    'prediction_rows': rows, 'loss_record': loss_path.name,
                    'loss_record_sha256': sha(loss_path)})
                save(out / f'case-{wi}-{ti}.json', record)
            run['cases'].append({'workload': name, 'task': task,
                'record': f'case-{wi}-{ti}.json', 'sha256': sha(out / f'case-{wi}-{ti}.json')})
            save(out / 'run.json', run)
        del data
        gc.collect()
    run['complete'] = True
    save(out / 'run.json', run)
    print('COMPLETE: 48 fresh banks, 240 native states; analyse prespecified outcomes', flush=True)


if __name__ == '__main__':
    main()
