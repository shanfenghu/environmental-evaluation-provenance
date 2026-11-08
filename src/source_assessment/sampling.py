"""Fixed cohort construction and disjoint, refreshed quota folds.

The dispersed cohort is an explicit supplied input. Northern-edge tasks and
protected cores follow their fixed identity/latitude rules; no losses tune them.
"""
import hashlib
import numpy as np

def refreshed_folds(cells,quota,count,seed):
    """Uniform quota marginals, disjoint within cycles, independent new cycles."""
    rng=np.random.default_rng(seed);cycle=min(len(ids)//quota[c] for c,ids in cells.items())
    for t in range(count):
        if t%cycle==0:permutations={c:rng.permutation(ids).tolist() for c,ids in cells.items()}
        position=t%cycle
        yield sorted(i for c,ids in permutations.items() for i in ids[position*quota[c]:(position+1)*quota[c]])


def design(data, task):
    """Return development/external/protected sets, eligible cells and floor quotas."""
    all_ids = set(range(len(data.meta)))
    if task == 'dispersed':
        external = all_ids-data.dev
    elif task == 'northern_edge':
        external = set()
        for c in sorted(data.cells):
            ids = [i for i in all_ids if data.meta[i]['cell'] == c]
            ordered = sorted(ids, key=lambda i: (-data.meta[i]['latitude'], data.meta[i]['id']))
            external.update(ordered[:int(.2*len(ids))])
    else:
        raise ValueError('Undeclared task')
    dev = all_ids-external
    protected, cells, quotas = set(), {}, {}
    for c in sorted(data.cells):
        ids = [i for i in dev if data.meta[i]['cell'] == c]
        ordered = sorted(ids, key=lambda i: hashlib.sha256(f"1041:{data.meta[i]['id']}".encode()).hexdigest())
        protected.update(ordered[:2])
        cells[c] = sorted(ordered[2:])
        quotas[c] = len(cells[c])//5
        assert quotas[c] > 0
    return dev, external, protected, cells, quotas
