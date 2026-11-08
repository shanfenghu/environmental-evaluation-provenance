"""Joint whole-cycle summaries, with descriptive small-sample bootstrap bands."""
import json
from pathlib import Path
import numpy as np
from .io import sha

def joint_summary(x, replicates, rng):
    shape = x.shape[1:]
    y = x.reshape(len(x), -1)
    mean = y.mean(axis=0)
    covariance = np.atleast_2d(np.cov(y, rowvar=False, ddof=1))/len(y)
    se = np.sqrt(np.maximum(0, np.diag(covariance)))
    active = (np.ptp(y, axis=0) > 0) & (se > 0)
    constant = ~active
    covariance[constant, :] = 0.
    covariance[:, constant] = 0.
    se[constant] = 0.
    samples = y[rng.integers(0, len(y), size=(replicates, len(y)))].mean(axis=1)
    critical = float(np.quantile(np.max(abs(samples[:, active]-mean[active])/se[active], axis=1), .95)) if active.any() else 0.
    return {'mean': mean.reshape(shape).tolist(), 'standard_error': se.reshape(shape).tolist(),
            'joint_band_low': (mean-critical*se).reshape(shape).tolist(),
            'joint_band_high': (mean+critical*se).reshape(shape).tolist(),
            'critical_max_standardised': critical, 'covariance_of_mean': covariance.tolist(),
            'constant_coordinates': np.flatnonzero(constant).tolist()}


def summarize(directory, config_path, output):
    """Regenerate all paired, crossed and bootstrap summaries from complete records."""
    cfg = json.loads(Path(config_path).read_text())
    out = Path(directory)
    run = json.loads((out/'run.json').read_text())
    assert run['complete'] and run['configuration_sha256'] == sha(config_path)
    assert run['loss_archive_sha256'] == sha(out/'losses.zip')
    summary = {'scope': 'Joint paired whole-cycle conditional MC uncertainty; descriptive bootstrap with eight cycles, no finite-sample or population coverage claim',
               'bootstrap_replicates': cfg['bootstrap_replicates'], 'cases': []}
    for ci, ref in enumerate(run['cases']):
        assert sha(out/ref['record']) == ref['sha256']
        case = json.loads((out/ref['record']).read_text())
        if len(case['cycles']) != cfg['cycles']:
            raise ValueError('A first-cycle replay cannot supply eight-cycle forecast/band summaries')
        rng = np.random.default_rng(cfg['bootstrap_seed']+ci)
        values = [c['means'] for c in case['cycles']]
        field_names = ['components', 'forward', 'reverse', 'shapley', 'interaction', 'gap',
                       'predicted_exclusion', 'swap_exclusion', 'swap_coupling']
        slices, arrays, position = {}, [], 0
        for field in field_names:
            a = np.array([v[field] for v in values]).reshape(cfg['cycles'], -1, len(run['columns']))
            slices[field] = [position, position+a.shape[1]]
            arrays.append(a)
            position += a.shape[1]
        all_values = np.concatenate(arrays, axis=1)
        joint = joint_summary(all_values, cfg['bootstrap_replicates'], rng)
        first = np.array([v['components'] for v in values[:4]])
        first_summary = joint_summary(first, cfg['bootstrap_replicates'], rng)
        predicted = np.array([v['predicted_exclusion'] for v in values[:4]])
        confirmed = np.array([v['swap_exclusion'] for v in values[4:]])
        pred_boot = predicted[rng.integers(0, 4, (cfg['bootstrap_replicates'], 4))].mean(axis=1)
        confirm_boot = confirmed[rng.integers(0, 4, (cfg['bootstrap_replicates'], 4))].mean(axis=1)
        delta = pred_boot-confirm_boot
        component_means = np.array(joint['mean'])[slice(*slices['components'])]
        dominance = np.argmax(abs(np.array([v['components'] for v in values])), axis=1)
        item = {'workload': case['workload'], 'task': case['task'], 'columns': run['columns'],
                'field_slices': slices, 'joint': joint,
                'first_four_components': first_summary,
                'cycle_dominance_counts': [[int(np.sum(dominance[:, j] == k)) for k in range(3)] for j in range(len(run['columns']))],
                'mean_dominant_component': np.argmax(abs(component_means), axis=0).tolist(),
                'followup': {'first_four_predicted_exclusion': predicted.mean(axis=0).tolist(),
                             'last_four_measured_exclusion': confirmed.mean(axis=0).tolist(),
                             'prediction_minus_confirmation': (predicted.mean(axis=0)-confirmed.mean(axis=0)).tolist(),
                             'difference_interval_low': np.quantile(delta, .025, axis=0).tolist(),
                             'difference_interval_high': np.quantile(delta, .975, axis=0).tolist()},
                'disde': case['disde'], 'target': case['target'],
                'elapsed_seconds_diagnostic_only': case['elapsed_seconds_diagnostic_only']}
        summary['cases'].append(item)
        print(case['workload'], case['task'], 'components', np.round(component_means, 6).tolist(), flush=True)
    Path(output).write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
