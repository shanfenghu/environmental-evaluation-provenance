"""Native fitting on exactly the selected earlier-response sources.

The scaler is refitted per allowed pool. Hydrological fitting uses log1p
responses, with the documented clipped inverse only when scoring predictions.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def fit_native(used,x,y,recipe):
    """Fit ridge or histogram boosting on source-indexed earlier arrays x/y.

    used selects all and only permitted source indices. Preprocessing is native
    to this fit; no full-population scaler is subtracted or reused.
    """
    used=sorted(used);xx=np.concatenate([x[i] for i in used]);yy=np.concatenate([y[i] for i in used])
    if recipe['kind']=='boosting':
        return HistGradientBoostingRegressor(**recipe['parameters'],random_state=41).fit(xx,yy)
    return make_pipeline(StandardScaler(),Ridge(alpha=recipe['alpha'],solver='cholesky')).fit(xx,yy)


def predict(model,x,hydrology):
    """Predict raw units, using the fixed nonnegative clipped inverse for flow."""
    value=model.predict(x)
    return np.maximum(0,np.expm1(np.clip(value,-20,20))) if hydrology else value


def fit_median(used, meta, y, month):
    """Raw-response medians by cell/month from permitted sources; months are 0–11."""
    table = {}
    for c in sorted({meta[i]['cell'] for i in used}):
        ids = sorted(i for i in used if meta[i]['cell'] == c)
        fallback = float(np.median(np.concatenate([y[i] for i in ids])))
        for m in range(12):
            values = np.concatenate([y[i][month[i] == m] for i in ids])
            table[c, m] = float(np.median(values)) if len(values) else fallback
    return table
