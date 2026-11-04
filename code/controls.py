"""Nonspatial controls for the prediction of held-out sharing.

Both ignore geometry. The first fits channel-specific recipient effects,
log lambda^(s)_ij = b_s + a^(s)_i + a^(s)_j, by the P objective; the second gives every pair the
channel's mean off-diagonal training count, so that each channel's share is constant.
"""

import numpy as np
from scipy.optimize import minimize

from reconstruct import OPTIONS


def recipient_effects_objective(v, C):
    """Poisson objective and gradient of channel-specific recipient effects, per training count."""
    S, n, _ = C.shape
    off = ~np.eye(n, dtype=bool)
    a, b = v[: S * n].reshape(S, n), v[S * n :]
    eta = b[:, None, None] + a[:, :, None] + a[:, None, :]
    exp_eta = np.exp(np.minimum(eta, 700.0))
    mean_term = exp_eta * (1.0 + np.maximum(eta - 700.0, 0.0))
    residual = (exp_eta - C) * off[None]
    mass = C[:, off].sum()
    value = np.sum((mean_term - C * eta)[:, off]) / mass
    return value, np.r_[(residual.sum(2) + residual.sum(1)).ravel(), residual.sum((1, 2))] / mass


def recipient_effects(C):
    """Channel-specific recipient effects fitted to the training counts C (S, n, n).

    Returns (parameters, record)."""
    S, n, _ = C.shape
    off = ~np.eye(n, dtype=bool)
    v0 = np.r_[np.zeros(S * n), np.log(np.maximum(C[:, off].mean(1), 0.001))]
    res = minimize(
        recipient_effects_objective,
        v0,
        args=(C,),
        jac=True,
        method='L-BFGS-B',
        options=OPTIONS,
    )
    value, _ = recipient_effects_objective(res.x, C)
    a, b = res.x[: S * n].reshape(S, n), res.x[S * n :]
    parameters = dict(node_effects_by_channel=a - a.mean(1, keepdims=True), intercepts=b + 2 * a.mean(1))
    return parameters, dict(
        objective=float(value * C[:, off].sum()), message=str(res.message), iterations=int(res.nit)
    )


def recipient_effects_log_mean(z):
    """Log means (S, n, n) of fitted recipient effects."""
    a = z['node_effects_by_channel']
    return z['intercepts'][:, None, None] + a[:, :, None] + a[:, None, :]


def constant(C):
    """A constant share per channel: every pair gets the channel's mean off-diagonal training count."""
    n = C.shape[1]
    off = ~np.eye(n, dtype=bool)
    intercepts = np.log(C[:, off].mean(1))
    eta = constant_log_mean(dict(intercepts=intercepts), n)
    return dict(intercepts=intercepts), dict(objective=float(np.sum((np.exp(eta) - C * eta)[:, off])))


def constant_log_mean(z, n):
    """Log means (S, n, n) of the constant control."""
    return np.broadcast_to(z['intercepts'][:, None, None], (len(z['intercepts']), n, n))
