"""Balanced same-task contrasts of a scored-block by excluded-block loss matrix.

Rows identify the fixed later scored block; columns identify the excluded earlier
block. Blocks share quotas/weights. Values must come from the declared source
roles; this numeric operation cannot certify those roles or infer causality.
"""
import numpy as np


def _matrix(values):
    q = np.asarray(values, dtype=float)
    if q.ndim < 2 or q.shape[0] != q.shape[1] or q.shape[0] < 2:
        raise ValueError('A square matrix of at least two scored/excluded blocks is required')
    if not np.isfinite(q).all():
        raise ValueError('All source-block losses must be finite')
    return q


def paired_audit(values, order=None):
    """Average own-exclusion minus successor-exclusion losses in a cyclic order.

    One global ordering applies across cells and joint learner coordinates. A
    loss-chosen ordering changes the experiment and is not a uniform-law sample.
    """
    q = _matrix(values)
    k = len(q)
    p = np.arange(k) if order is None else np.asarray(order)
    if p.shape != (k,) or not np.issubdtype(p.dtype, np.integer) or set(p) != set(range(k)):
        raise ValueError('Order must be a permutation of all block indices')
    return np.mean(q[p, p]-q[p, np.roll(p, -1)], axis=0)


def averaged_audit(values):
    """Average the audit over every distinct other fit, preserving its target.

    Conditional on unordered disjoint blocks and their fixed fitting outcomes,
    this equals the mean cyclic audit under a uniformly random global block
    ordering. Its unconditional covariance is no larger in positive-semidefinite
    order, by total covariance. It removes pairing variation, not block selection
    or population uncertainty. Scoring may require more work than a single cycle.
    """
    q = _matrix(values)
    k = len(q)
    diagonal = q[np.arange(k), np.arange(k)].sum(axis=0)
    return diagonal/k-(q.sum(axis=(0, 1))-diagonal)/(k*(k-1))


def interaction_matrix(values):
    """Remove additive scored-block and fitted-state effects by double centring.

    Balanced audit contrasts are unchanged. Remaining effects can include pool
    geometry, row counts or source-history interactions; no cause is isolated.
    """
    q = _matrix(values)
    return q-q.mean(axis=0, keepdims=True)-q.mean(axis=1, keepdims=True)+q.mean(axis=(0, 1), keepdims=True)
