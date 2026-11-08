"""Source-role rejection and an exhaustive disjoint-source-swap oracle."""
import itertools
import unittest
import numpy as np
from source_assessment.queries import FitScores, ScoreQuery
from source_assessment.queries import cell_score


class EvaluationStateTests(unittest.TestCase):
    def test_wrong_source_version_and_forbidden_holdout_queries_are_rejected(self):
        state = FitScores('v1', 'recipe', (0, 1), (0, 1, 2), np.ones((3, 3)))
        for version, fit, score, role, external in [
            ('v2', (0, 1), (2,), 'entity_holdout', {2}),
            ('v1', (0,), (2,), 'entity_holdout', {2}),
            ('v1', (0, 1), (1,), 'entity_holdout', {2}),
            ('v1', (0, 1), (2,), 'external_holdout', {1, 2})]:
            with self.assertRaises(ValueError):
                ScoreQuery('q', version, 'recipe', fit, score, role).evaluate(state, {0: [0, 1, 2]}, external)
        value, _ = ScoreQuery('temporal', 'v1', 'recipe', (0, 1), (0,),
                              'temporal_reference').evaluate(state, {0: [0, 1, 2]}, {2})
        np.testing.assert_array_equal(value, np.ones(3))

    def test_disjoint_equal_size_swap_recovers_coupling_with_unequal_inclusion_rates(self):
        cells, quotas = {0: [0, 1, 2], 1: [3, 4, 5, 6]}, {0: 1, 1: 2}
        subsets = lambda ids, q: list(itertools.combinations(ids, q))
        pairs, coupling = [], []
        for aa, bb in itertools.product(subsets(cells[0], 1), subsets(cells[1], 2)):
            v = set(aa+bb)
            losses = lambda mask: {i: float(i+.7*(i in mask)+.3*((i+1)%7 in mask)) for i in range(7)}
            own = losses(v)
            coupling.append(cell_score(own, cells, v)-cell_score(own, cells, range(7)))
            for cc, dd in itertools.product(subsets(set(cells[0])-v, 1), subsets(set(cells[1])-v, 2)):
                w = set(cc+dd)
                other = losses(w)
                pairs.append(sum((1-quotas[c]/len(ids))*np.mean([own[i]-other[i] for i in v & set(ids)])
                                 for c, ids in cells.items())/len(cells))
        self.assertAlmostEqual(np.mean(pairs), np.mean(coupling), places=14)


if __name__ == '__main__':
    unittest.main()
