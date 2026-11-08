"""Native loss-vector execution and a fixed-model overlap diagnostic."""
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .models import fit_native, predict, fit_median
from .queries import FitScores, expand

def fitted_state(data, used, external, scored, recipes, recipe_version):
    """Fit three permitted recipes, score later rows and retain per-source MAEs.

    No fitted model pickle is required after scoring; timings are diagnostics.
    """
    if set(used) & set(external):
        raise ValueError('External predictive training sources')
    start = time.perf_counter()
    # No future response is passed into either native fitting call.
    models = [fit_native(used, data.x, data.y, recipes[name]) for name in ['ridge', 'boosting']]
    table = fit_median(used, data.meta, data.raw, data.month)
    fit_seconds = time.perf_counter()-start
    start = time.perf_counter()
    xx = np.concatenate([data.z[i] for i in scored])
    predictions = [predict(m, xx, data.hydro) for m in models]
    values = []
    offset = 0
    for i in scored:
        end = offset+len(data.w[i])
        med = np.array([table[data.meta[i]['cell'], int(month)] for month in data.future[i]])
        values.append([float(np.mean(abs(p[offset:end]-data.w[i]))) for p in predictions]
                      + [float(np.mean(abs(med-data.w[i])))])
        offset = end
    losses = np.asarray(values)
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite native losses')
    return (FitScores(data.input_hash, recipe_version, tuple(sorted(used)), tuple(scored), losses),
            {'fit_seconds': fit_seconds, 'score_seconds': time.perf_counter()-start,
             'training_rows': sum(len(data.y[i]) for i in used)})


def cell_values(state, cells, selected):
    """Return source means per cell for a valid nonempty declared scored set."""
    lookup = {i: j for j, i in enumerate(state.scored_sources)}
    return np.array([np.mean([state.losses[lookup[i]] for i in sorted(set(selected)&set(ids))], axis=0)
                     for ids in cells.values()])


def disde(data, cells, external, full, seed):
    """Normalised overlap weighting per cell; no population/causal CI is imported."""
    lookup = {i: j for j, i in enumerate(full.scored_sources)}
    records, parts = [], []
    for c, source in cells.items():
        target = sorted(i for i in external if data.meta[i]['cell'] == c)
        ids = source+target
        labels = np.array([0]*len(source)+[1]*len(target))
        features = np.array([data.x[i].mean(axis=0) for i in ids])
        probabilities = np.zeros(len(ids))
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed+c).split(features, labels):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000))
            model.fit(features[train], labels[train])
            probabilities[test] = model.predict_proba(features[test])[:, 1]
        probabilities = np.clip(probabilities, 1e-6, 1-1e-6)
        alpha = float(labels.mean())
        denominator = (1-alpha)*probabilities+alpha*(1-probabilities)
        wp = probabilities[:len(source)]/denominator[:len(source)]
        wq = (1-probabilities[len(source):])/denominator[len(source):]
        lp = np.array([full.losses[lookup[i]] for i in source])
        lq = np.array([full.losses[lookup[i]] for i in target])
        sp, sq = np.average(lp, weights=wp, axis=0), np.average(lq, weights=wq, axis=0)
        parts.append(np.array([lp.mean(axis=0)-sp, sp-sq, sq-lq.mean(axis=0)]))
        records.append({'cell': c, 'source': source, 'target': target, 'alpha': alpha,
                        'oof_probability': probabilities.tolist(), 'source_weights': wp.tolist(),
                        'target_weights': wq.tolist(),
                        'source_effective_n': float(wp.sum()**2/(wp@wp)),
                        'target_effective_n': float(wq.sum()**2/(wq@wq))})
    return {'components': expand(np.mean(parts, axis=0)).tolist(), 'cells': records,
            'scope': 'Fixed predictive model; normalised overlap plug-in conditional on observed cells and earlier covariate means; no causal/population interpretation'}
