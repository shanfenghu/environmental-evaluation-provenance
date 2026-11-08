"""A native fixture with unreadable external fitting responses."""
from types import SimpleNamespace
import unittest
import numpy as np
from source_assessment.native import fitted_state


class UnreadableResponse:
    def __array__(self, *args, **kwargs):
        raise AssertionError('External fitting response was read')


class NativeRoundOracleTests(unittest.TestCase):
    def test_future_scoring_does_not_read_external_earlier_responses(self):
        rng = np.random.default_rng(513)
        x = [rng.normal(size=(36, 3)) for _ in range(4)]
        y = [a[:, 0]+.2*a[:, 1] for a in x[:3]]+[UnreadableResponse()]
        raw = y.copy()
        z = [rng.normal(size=(12, 3)) for _ in range(4)]
        d = SimpleNamespace(x=x, y=y, raw=raw, z=z, w=[a[:, 0] for a in z],
                            month=[np.arange(36)%12 for _ in x],
                            future=[np.arange(12) for _ in x],
                            meta=[{'cell': 0} for _ in x], hydro=False, input_hash='fixture')
        recipes = {'ridge': {'kind': 'ridge', 'alpha': 100.},
                   'boosting': {'kind': 'boosting', 'parameters': {'loss': 'absolute_error',
                                'max_iter': 100, 'early_stopping': False, 'min_samples_leaf': 20}}}
        state, _ = fitted_state(d, {0, 1, 2}, {3}, [3], recipes, 'fixture')
        self.assertEqual(state.fitted_sources, (0, 1, 2))
        self.assertTrue(np.isfinite(state.losses).all())
        with self.assertRaises(ValueError):
            fitted_state(d, {0, 3}, {3}, [3], recipes, 'fixture')


if __name__ == '__main__':
    unittest.main()
