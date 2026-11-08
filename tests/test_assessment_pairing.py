"""Finite-order oracles for balanced source-block comparisons."""
import itertools
import unittest
import numpy as np
from source_assessment.pairing import averaged_audit, interaction_matrix, paired_audit


class PairingTests(unittest.TestCase):
    def test_two_blocks_have_no_order_variation_for_joint_contrasts(self):
        # The only two orders are rotations, even for nonadditive joint losses.
        losses = np.array([[[2., 7.], [10., -1.]],
                           [[4., 8.], [0., 4.]]])
        expected = np.array([-6., 2.])
        for order in [(0, 1), (1, 0)]:
            np.testing.assert_array_equal(paired_audit(losses, order), expected)
        np.testing.assert_array_equal(averaged_audit(losses), expected)

    def test_all_orders_and_total_covariance_match_exactly(self):
        rng = np.random.default_rng(103)
        # Two distinct unordered block banks with joint, dependent learner losses.
        banks = [rng.normal(size=(5, 5, 3)) for _ in range(2)]
        outcomes = []
        means = []
        conditional = []
        for q in banks:
            x = np.array([paired_audit(q, (0,)+p) for p in itertools.permutations(range(1, 5))])
            np.testing.assert_allclose(averaged_audit(q), x.mean(axis=0), atol=1e-14)
            means.append(x.mean(axis=0));outcomes.extend(x)
            conditional.append(np.cov(x, rowvar=False, bias=True))
        total = np.cov(outcomes, rowvar=False, bias=True)
        between = np.cov(means, rowvar=False, bias=True)
        np.testing.assert_allclose(total-between, np.mean(conditional, axis=0), atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(total-between).min(), -1e-14)

    def test_additive_pool_and_cohort_effects_cancel_but_interaction_remains(self):
        k = 5
        additive = np.arange(k)[:, None]**2 + 100*np.arange(k)[None, :]
        interaction = 2*np.eye(k)
        for p in itertools.permutations(range(k)):
            self.assertAlmostEqual(float(paired_audit(additive, p)), 0)
            self.assertAlmostEqual(float(paired_audit(additive+interaction, p)), 2)
        self.assertAlmostEqual(float(averaged_audit(additive+interaction)), 2)
        q = additive+interaction
        self.assertAlmostEqual(float(averaged_audit(interaction_matrix(q))), 2)
        self.assertAlmostEqual(float(k/(k-1)*np.diag(interaction_matrix(q)).mean()), 2)
        # A fixed prediction rule has identical columns, hence exactly zero.
        self.assertAlmostEqual(float(averaged_audit(np.repeat(np.arange(k)[:, None], k, axis=1))), 0)

    def test_invalid_matrix_or_order_is_rejected(self):
        for q in [np.ones(5), np.ones((3, 4)), np.ones((1, 1)), np.full((2, 2), np.nan)]:
            with self.assertRaises(ValueError):averaged_audit(q)
        for p in [[0, 0], [0.0, 1.0], [0], [0, 2]]:
            with self.assertRaises(ValueError):paired_audit(np.ones((2, 2)), p)


if __name__ == '__main__':unittest.main()
