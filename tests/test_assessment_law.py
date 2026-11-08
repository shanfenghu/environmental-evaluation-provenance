"""Exhaustive quota-law oracle for the coupling interpretation and score weights."""
import itertools
import unittest
import numpy as np
from source_assessment.queries import cell_score, components


class ValidationGapTests(unittest.TestCase):
    def test_exact_quota_law_covariance_matches_expected_coupling(self):
        cells = {0: [0, 1, 2], 1: [3, 4, 5, 6]}
        quotas = {0: 1, 1: 2}
        assignments = [set(a+b) for a, b in itertools.product(
            itertools.combinations(cells[0], 1), itertools.combinations(cells[1], 2))]
        inclusion, losses, coupling = [], [], []
        for v in assignments:
            # An exact finite response deliberately depends on self and partner exclusion.
            g = {i: float(i+.7*(i in v)+.3*((i+1)%7 in v)) for i in range(7)}
            h = cell_score(g, cells, v)
            population = cell_score(g, cells, range(7))
            coupling.append(components(h, population, 0., 1.)[0])
            inclusion.append([i in v for i in range(7)])
            losses.append([g[i] for i in range(7)])
        inc, g = np.array(inclusion), np.array(losses)
        cov = np.mean(inc*g, axis=0)-inc.mean(axis=0)*g.mean(axis=0)
        weighted = sum(cov[i]/(2*quotas[c]) for c, ids in cells.items() for i in ids)
        self.assertAlmostEqual(np.mean(coupling), weighted, places=14)
        self.assertGreater(abs(weighted), .1)

    def test_fixed_fitted_loss_has_zero_expected_coupling(self):
        cells = {0: [0, 1, 2, 3]}
        g = {i: np.array([float(i), float(i*i)]) for i in range(4)}
        values = [cell_score(g, cells, v)-cell_score(g, cells, range(4))
                  for v in itertools.combinations(range(4), 2)]
        np.testing.assert_allclose(np.mean(values, axis=0), 0., atol=1e-15)

    def test_unequal_cells_use_equal_cell_weights_and_reject_missing_roles(self):
        g = {0: 0., 1: 10., 2: 10., 3: 10.}
        self.assertEqual(cell_score(g, {0: [0], 1: [1, 2, 3]}, range(4)), 5.)
        for cells, selected in [({0: [0], 1: [1]}, [0]),
                                ({0: [0, 1], 1: [1, 2]}, [0, 1, 2]),
                                ({0: [0]}, [0, 3])]:
            with self.assertRaises(ValueError):
                cell_score(g, cells, selected)


if __name__ == '__main__':
    unittest.main()
