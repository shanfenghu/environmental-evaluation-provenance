"""Bind scores to fitting/scoring sets, versions, roles and equal-cell weights.

Loss rows contain three finite learner losses. Guards concern declared local
states, not unobserved upstream lineage or environmental independence.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

def cell_score(losses, cells, selected):
    """Average days within each supplied entity loss, then entities/cells equally."""
    selected = set(selected)
    groups = [selected.intersection(ids) for ids in cells.values()]
    if not groups or any(not ids for ids in groups):
        raise ValueError('Every scored cell must be represented')
    if set().union(*groups) != selected or sum(map(len, groups)) != len(selected):
        raise ValueError('Selected entities must belong to a disjoint cell partition')
    return np.mean([np.mean([losses[i] for i in sorted(ids)], axis=0)
                    for ids in groups], axis=0)


def components(validation, pool_population, full_population, target):
    """Return coupling/refit/cohort contrasts in a specified reference order.

    Only the expected first term has the inclusion/loss covariance interpretation;
    a single assignment also contains ordinary score-sampling fluctuation.
    """
    h, a, full, q = map(np.asarray, [validation, pool_population, full_population, target])
    if not all(x.shape == h.shape and np.isfinite(x).all() for x in [h, a, full, q]):
        raise ValueError('Finite matching score shapes required')
    return np.stack([h-a, a-full, full-q])


def digest(value):
    """Canonical JSON fingerprint for a declared local state or score query."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class FitScores:
    """Source-indexed finite losses, with exact fit/input/recipe identities."""
    input_version: str
    recipe_version: str
    fitted_sources: tuple
    scored_sources: tuple
    losses: np.ndarray


@dataclass(frozen=True)
class ScoreQuery:
    """A scored set and role bound to one explicit fitted source state."""
    name: str
    input_version: str
    recipe_version: str
    fitted_sources: tuple
    scored_sources: tuple
    role: str

    def evaluate(self, state, cells, external):
        """Validate declared source permissions and return score plus query digest."""
        if (self.input_version != state.input_version or self.recipe_version != state.recipe_version
                or self.fitted_sources != state.fitted_sources):
            raise ValueError('Query does not match the fitted source/input/recipe state')
        fit, score, ext = set(self.fitted_sources), set(self.scored_sources), set(external)
        if len(fit) != len(self.fitted_sources) or len(score) != len(self.scored_sources):
            raise ValueError('Distinct fitted and scored identities required')
        if fit & ext:
            raise ValueError('External prediction responses cannot enter predictive fitting')
        if self.role in ['entity_holdout', 'external_holdout']:
            if fit & score:
                raise ValueError('Held-out query has fitted/scored source overlap')
            if self.role == 'external_holdout' and not score <= ext:
                raise ValueError('External query must score only external sources')
        elif self.role != 'temporal_reference':
            raise ValueError('Unknown score role')
        index = {i: j for j, i in enumerate(state.scored_sources)}
        if not score <= index.keys():
            raise ValueError('Missing source losses')
        value = cell_score({i: state.losses[index[i]] for i in score}, cells, score)
        return value, digest({'input': self.input_version, 'recipe': self.recipe_version,
            'fit': self.fitted_sources, 'score': self.scored_sources,
            'role': self.role, 'cells': cells, 'weights': 'equal_cell_entity_day'})


def query(name, state, scored, role, cells, external):
    """Evaluate an equal-cell score with explicit sources and assessment role."""
    obj = ScoreQuery(name, state.input_version, state.recipe_version,
                     state.fitted_sources, tuple(sorted(scored)), role)
    return obj.evaluate(state, cells, external)


def expand(raw):
    """Retain covariance by deriving all contrasts from the same three raw losses."""
    r = np.asarray(raw)
    return np.stack([r[..., 0], r[..., 1], r[..., 2], r[..., 0]-r[..., 2],
                     r[..., 1]-r[..., 2], r[..., 1]-r[..., 0]], axis=-1)


def crossed(h, a, f, q, l, m):
    """Known telescoping and two-factor Shapley allocations, with both orders."""
    return {'components': np.stack([h-a, a-f, f-q]),
            'forward': np.stack([h-l, l-q]),
            'reverse': np.stack([m-q, h-m]),
            'shapley': np.stack([.5*((h-l)+(m-q)), .5*((l-q)+(h-m))]),
            'interaction': (h-l)-(m-q), 'gap': h-q}
