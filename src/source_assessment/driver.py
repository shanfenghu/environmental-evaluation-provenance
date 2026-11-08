"""Execute the frozen native recipe, optionally its first recorded cycle only.

The reduced stage exercises both target tasks in both domains. It is a fixed
reproduction check, not additional assignments or an accuracy/tuning exercise.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import sklearn
from .io import sha, save
from .data import Data
from .sampling import refreshed_folds, design
from .queries import digest, expand, query, crossed
from .native import fitted_state, cell_values, disde


def run(config_path, data_dir, cohort_path, output, fixed_cycle=False, full_only=False):
    """Recreate native fitted states and all declared scores at a new destination.

    Uses source-indexed losses instead of requiring model-pickle archives.
    Existing output directories are rejected; threads must be fixed before NumPy
    is imported. The supplied frozen configuration defines the full eight-cycle
    recipe; fixed_cycle replays its first cycle, while full_only replays only
    the full fitted state in every task. Neither is a fresh scientific study.
    """
    assert all(os.environ.get(n) == '1' for n in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'])
    if fixed_cycle and full_only:
        raise ValueError('Choose one reduced replay stage')
    cfg = json.loads(Path(config_path).read_text())
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    source = Path(__file__).parent
    run = {'complete': False, 'stage': 'full_only' if full_only else ('fixed_cycle' if fixed_cycle else 'full'),
           'cycle_indices': [] if full_only else ([0] if fixed_cycle else list(range(cfg['cycles']))),
           'source_sha256': {p.name: sha(p) for p in sorted(source.glob('*.py'))},
           'configuration_sha256': sha(config_path), 'cohort_sha256': sha(cohort_path),
           'environment_lock_sha256': sha(Path(config_path).parent.parent/'requirements.lock'),
           'environment': {'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
           'columns': ['ridge', 'boosting', 'seasonal', 'ridge_minus_seasonal', 'boosting_minus_seasonal', 'boosting_minus_ridge'],
           'cases': [], 'scope': 'Conditional assessment on retained populations; fixed replay is not new scientific evidence or a new-population/causal guarantee'}
    save(out/'run.json', run)
    loss_archive = {}
    cohorts = json.loads(Path(cohort_path).read_text())
    for wi, name in enumerate(cfg['workloads']):
        data = Data(name, data_dir, cohorts)
        with np.load(data.path) as arrays:
            end = max(int(np.max(arrays[f't{i}'])) for i in range(len(data.meta)))
            begin = min(int(np.min(arrays[f's{i}'])) for i in range(len(data.meta)))
        assert end < begin
        for ti, task in enumerate(cfg['tasks']):
            started = time.perf_counter()
            dev, external, protected, cells, quotas = design(data, task)
            eligible = set().union(*map(set, cells.values()))
            scored = sorted(eligible|external)
            all_cells = {c: [i for i in scored if data.meta[i]['cell'] == c] for c in cells}
            recipe_version = digest({'learners': cfg['learners'], 'seasonal': 'cell_month_median',
                                      'response_transform': 'native_log1p_clip_inverse' if data.hydro else 'identity'})
            full, full_timing = fitted_state(data, dev, external, scored, cfg['learners'], recipe_version)
            full_key = f'{wi}_{ti}_full'
            loss_archive[full_key] = full.losses
            f, fsig = query('F', full, eligible, 'temporal_reference', cells, external)
            q, qsig = query('Q', full, external, 'external_holdout', all_cells, external)
            case = {'workload': name, 'task': task, 'input_sha256': data.input_hash,
                    'metadata_sha256': sha(data.path.with_suffix('.json')),
                    'development': sorted(dev), 'external': sorted(external), 'protected': sorted(protected),
                    'cells': cells, 'quotas': quotas, 'scored': scored, 'recipe_version': recipe_version,
                    'latest_training_date': str(np.datetime64(end, 'D')), 'earliest_scoring_date': str(np.datetime64(begin, 'D')),
                    'full_loss_key': full_key, 'full_timing': full_timing,
                    'full_population': expand(f).tolist(), 'target': expand(q).tolist(),
                    'full_query_signatures': {'F': fsig, 'Q': qsig},
                    'disde': disde(data, cells, external, full, cfg['seed']+wi*100000+ti*10000+901), 'cycles': []}
            pi = np.array([quotas[c]/len(ids) for c, ids in cells.items()])
            for cycle in run['cycle_indices']:
                seed = cfg['seed']+wi*100000+ti*10000+cycle
                masks = list(refreshed_folds(cells, quotas, cfg['folds'], seed))
                assert len(set().union(*map(set, masks))) == sum(map(len, masks))
                states, fold_rows = [], []
                for j, v in enumerate(masks):
                    used = dev-set(v)
                    assert protected <= used
                    state, timing = fitted_state(data, used, external, scored, cfg['learners'], recipe_version)
                    key = f'{wi}_{ti}_{cycle}_{j}'
                    loss_archive[key] = state.losses
                    states.append(state)
                    h, hs = query('H', state, v, 'entity_holdout', cells, external)
                    a, ass = query('A', state, eligible, 'temporal_reference', cells, external)
                    l, ls = query('L', full, v, 'temporal_reference', cells, external)
                    m, ms = query('M', state, external, 'external_holdout', all_cells, external)
                    values = crossed(h, a, f, q, l, m)
                    assert np.max(abs(values['components'].sum(axis=0)-values['gap'])) < 1e-10
                    cc = cell_values(state, cells, v)-cell_values(state, cells, eligible)
                    values['predicted_exclusion'] = np.mean(cc/(1-pi[:, None]), axis=0)
                    fold_rows.append({'validation': v, 'loss_key': key, 'timing': timing,
                        'scores': {k: expand(z).tolist() for k, z in [('H', h), ('A', a), ('L', l), ('M', m)]},
                        'values': {k: expand(z).tolist() for k, z in values.items()},
                        'query_signatures': {'H': hs, 'A': ass, 'L': ls, 'M': ms}})
                    print('Fit', name, task, cycle+1, j+1, '/', cfg['folds'], flush=True)
                for j, v in enumerate(masks):
                    own = cell_values(states[j], cells, v)
                    other = cell_values(states[(j+1)%cfg['folds']], cells, v)
                    _, sig = query('matched_swap', states[(j+1)%cfg['folds']], v,
                                   'temporal_reference', cells, external)
                    fold_rows[j]['query_signatures']['matched_swap'] = sig
                    fold_rows[j]['values']['swap_exclusion'] = expand(np.mean(own-other, axis=0)).tolist()
                    fold_rows[j]['values']['swap_coupling'] = expand(np.mean((1-pi[:, None])*(own-other), axis=0)).tolist()
                    fold_rows[j]['swap_training_rows'] = fold_rows[(j+1)%cfg['folds']]['timing']['training_rows']
                fields = fold_rows[0]['values']
                means = {k: np.mean([row['values'][k] for row in fold_rows], axis=0).tolist() for k in fields}
                case['cycles'].append({'cycle': cycle, 'seed': seed, 'folds': fold_rows, 'means': means})
                save(out/f'case-{wi}-{ti}.json', case)
            case['elapsed_seconds_diagnostic_only'] = time.perf_counter()-started
            save(out/f'case-{wi}-{ti}.json', case)
            run['cases'].append({'record': f'case-{wi}-{ti}.json', 'workload': name, 'task': task, 'sha256': sha(out/f'case-{wi}-{ti}.json')})
            with (out/'losses.zip').open('wb') as stream:
                np.savez_compressed(stream, **loss_archive)
            save(out/'run.json', run)
            print('Complete task', name, task, case['elapsed_seconds_diagnostic_only'], flush=True)
            del full
            if case['cycles']: del states
            gc.collect()
        del data
        gc.collect()
    run['complete'] = True
    run['loss_archive_sha256'] = sha(out/'losses.zip')
    run['loss_archive_keys'] = sorted(loss_archive)
    save(out/'run.json', run)
