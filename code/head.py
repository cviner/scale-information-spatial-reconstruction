"""Labelled calibration head for a fixed shape.

The shape is centred and scaled to a mean squared pair distance of 900; the head then fits
recipient effects, one intercept per channel and one global squared scale t >= 0 under the P mean,
eta = b_s + a_i + a_j - t d_ij^2 / (2 sigma_s^2), to the training counts. It can change the size of a
map and each recipient's loading, but cannot move recipients relative to one another.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from reconstruct import OPTIONS


def head_objective(v, counts, scaled_distance, off):
    """P objective per training count of a fixed shape with free recipient effects, intercepts and scale."""
    n, k = len(off), len(counts)
    a, b, t = v[:n], v[n : n + k], v[-1]
    eta = b[:, None, None] + a[None, :, None] + a[None, None, :] - t * scaled_distance
    exp_eta = np.exp(np.minimum(eta, 700.0))
    mean_term = exp_eta * (1.0 + np.maximum(eta - 700.0, 0.0))
    residual = (exp_eta - counts) * off[None]
    mass = counts[:, off].sum()
    value = ((mean_term - counts * eta) * off[None]).sum() / mass
    grad_a = residual.sum((0, 2)) + residual.sum((0, 1))
    grad_b = residual.sum((1, 2))
    grad_t = -np.sum(residual * scaled_distance)
    return value, np.r_[grad_a, grad_b, grad_t] / mass


def calibrate(x, counts, widths):
    """Calibrate a fixed shape to the training counts; returns (calibrated map, optimizer record)."""
    x = np.asarray(x, dtype=float).copy()
    x -= x.mean(0)
    pair2 = pdist(x, 'sqeuclidean')
    if not np.isfinite(pair2).all() or pair2.mean() <= 0:
        raise ValueError('Nonfinite or constant uncalibrated map')
    x *= np.sqrt(900 / pair2.mean())
    n, k = len(x), len(counts)
    off = ~np.eye(n, dtype=bool)
    scaled_distance = squareform(pdist(x, 'sqeuclidean'))[None] / (2 * widths[:, None, None])
    start = np.r_[np.zeros(n), np.log(np.maximum(counts[:, off].mean(1), 0.001)), 1.0]
    res = minimize(
        head_objective,
        start,
        args=(counts, scaled_distance, off),
        jac=True,
        method='L-BFGS-B',
        bounds=[(None, None)] * (n + k) + [(0, None)],
        options=OPTIONS,
    )
    value, _ = head_objective(res.x, counts, scaled_distance, off)
    a, b, t = res.x[:n], res.x[n : n + k], float(res.x[-1])
    return dict(
        coordinates=x * np.sqrt(t),
        node_effects=a - a.mean(),
        intercepts=b + 2 * a.mean(),
        widths=widths,
        normalized_shape=x,
        fitted_squared_scale=np.array(t),
    ), dict(objective=float(value * counts[:, off].sum()), message=str(res.message), iterations=int(res.nit))
