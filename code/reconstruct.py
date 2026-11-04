"""Joint P, fitted to the labelled channels, and its pooled competitor on planar recipients.

Widths are sigma_s^2 = 2 r_s^2. Both objectives are divided by the total off-diagonal training
counts, coordinates are optimized in units of 480, and L-BFGS-B runs for at most 12,000
iterations and 24,000 evaluations with ftol 1e-14 and gtol 1e-8. Six starts are computed from
the training counts, and the start with the lowest training objective is kept.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import msld
from graph import classical_mds

UNIT = 480
OPTIONS = dict(maxiter=12000, maxfun=24000, ftol=1e-14, gtol=1e-8)


def starts(train, gx, g_indices, seed):
    """G's map if G covers every recipient, otherwise classical MDS of sqrt(-2 r_max^2 log c);
    that MDS map scaled by 0.5, 1 and 2; and two Gaussian maps scaled to the norms of the MDS map and
    of the first start."""
    n = len(train['cosine'])
    cmds = classical_mds(np.sqrt(-max(train['widths']) * np.log(np.clip(train['cosine'], 1e-6, 1))))
    first = gx if len(g_indices) == n else cmds

    def gaussian(offset, like):
        x = np.random.RandomState(seed + offset).normal(size=(n, 2))
        x -= x.mean(0)
        x *= np.linalg.norm(like - like.mean(0)) / np.linalg.norm(x)
        return x

    out = np.array([first, 0.5 * cmds, cmds, 2 * cmds, gaussian(91, cmds), gaussian(92, first)])
    out -= out.mean(1, keepdims=True)
    return out


def profiled(theta, C, widths, totals, n, off, return_state=False):
    """Pooled objective sum_{i!=j} [Lambda - T log Lambda] with Lambda = sum_s lambda^(s), where each
    intercept is profiled so that the channel's predicted total equals its training total."""
    x = theta[: 2 * n].reshape(n, 2)
    a = theta[2 * n :]
    delta = x[:, None] - x[None]
    d2 = np.square(delta).sum(-1)
    u = a[None, :, None] + a[None, None, :] - d2[None] / (2 * widths[:, None, None])
    b = np.log(totals) - logsumexp(u[:, off], axis=1)
    loglam = u + b[:, None, None]
    logmu = logsumexp(loglam, axis=0)
    lam = np.exp(np.where(off[None], loglam, -np.inf))
    mu = lam.sum(0)
    shares = np.exp(loglam - logmu[None])
    delta_eta = (lam - C[None] * shares) * off[None]
    delta_eta -= lam / totals[:, None, None] * delta_eta.sum((1, 2))[:, None, None]
    loss = float(np.sum((mu - C * logmu)[off]))
    ga = 2 * delta_eta.sum((0, 2))
    gx = -2 * ((delta_eta / widths[:, None, None]).sum(0)[:, :, None] * delta).sum(1)
    if return_state:
        return x, a, b, lam
    return loss, np.r_[gx.ravel(), ga]


def pooled_objective(c, widths, n, off):
    """Pooled objective and gradient per training count as functions of v = theta / units; returns
    (objective, units), with the coordinates in units of UNIT."""
    total, totals = c.sum(0), c[:, off].sum(1)
    mass = totals.sum()
    units = np.ones(3 * n)
    units[: 2 * n] = UNIT

    def objective(v):
        f, g = profiled(v * units, total, widths, totals, n, off)
        return f / mass, g * units / mass

    return objective, units


def fit(c, widths, start_coordinates, method='labelled'):
    """Fit joint P (method='labelled') or the pooled competitor (method='pooled') from every start.

    Returns (states, records, chosen): states[i] = (x, a, b) after centring the coordinates and the
    recipient effects, and chosen is the start with the lowest training objective."""
    S = len(widths)
    n = len(c[0])
    off = ~np.eye(n, dtype=bool)
    totals = c[:, off].sum(1)
    mass = totals.sum()
    if method == 'labelled':
        objective, units = msld.scaled_objective(c, widths, n, normalizer=mass, coordinate_unit=UNIT, off=off)
        b0 = msld.initial_intercepts(c, off)
    elif method == 'pooled':
        objective, units = pooled_objective(c, widths, n, off)
    else:
        raise ValueError(method)
    states, records = [], []
    for i, x0 in enumerate(start_coordinates):
        start = msld.pack(x0, np.zeros(n), b0) if method == 'labelled' else np.r_[x0.ravel(), np.zeros(n)]
        res = minimize(objective, start / units, jac=True, method='L-BFGS-B', options=OPTIONS)
        f, _ = objective(res.x)
        if method == 'labelled':
            x, a, b = msld.unpack(res.x * units, n, S, 2)
        else:
            x, a, b, _ = profiled(res.x * units, c.sum(0), widths, totals, n, off, return_state=True)
        states.append(msld.gauge(x, a, b))
        records.append(
            dict(start=i, objective=float(f * mass), message=str(res.message), iterations=int(res.nit))
        )
    chosen = int(np.argmin([q['objective'] for q in records]))
    return states, records, chosen


def continue_pooled(c, widths, pooled_state, labelled_states):
    """Two training-only continuations of the pooled fit: from its own selected state, and from the
    joint-P start state with the lowest pooled objective. The lowest of the fit before continuation and
    the two continuations is kept. Returns (state, records, chosen, details): chosen is
    'before_continuation', 0 or 1, and details holds both continuation states, the joint-P start used,
    the pooled objective at every joint-P start, and the pooled objective before continuation."""
    n = c.shape[1]
    off = ~np.eye(n, dtype=bool)
    total, totals = c.sum(0), c[:, off].sum(1)
    mass = totals.sum()
    objective, units = pooled_objective(c, widths, n, off)
    labelled = [np.r_[x.ravel(), a] for x, a, _ in labelled_states]
    values = [float(profiled(v, total, widths, totals, n, off)[0]) for v in labelled]
    labelled_start = int(np.argmin(values))
    initials = [np.r_[pooled_state[0].ravel(), pooled_state[1]], labelled[labelled_start]]
    records, states = [], []
    for k, initial in enumerate(initials):
        res = minimize(objective, initial / units, jac=True, method='L-BFGS-B', options=OPTIONS)
        value, _ = objective(res.x)
        x, a, b, _ = profiled(res.x * units, total, widths, totals, n, off, return_state=True)
        states.append(msld.gauge(x, a, b))
        records.append(
            dict(start=k, objective=float(value * mass), message=str(res.message), iterations=int(res.nit))
        )
    before_value, _ = profiled(initials[0], total, widths, totals, n, off)
    best = min(
        [*records, dict(start='before_continuation', objective=float(before_value))],
        key=lambda q: q['objective'],
    )
    state = pooled_state if best['start'] == 'before_continuation' else states[best['start']]
    details = dict(
        states=states,
        labelled_start=labelled_start,
        pooled_objective_at_labelled_starts=values,
        objective_before_continuation=float(before_value),
    )
    return state, records, best['start'], details
