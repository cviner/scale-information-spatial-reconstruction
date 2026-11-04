"""Geometric, predictive and biological endpoints.

Every map is aligned to the reference by one least-squares similarity (scale, rotation, reflection
and translation) for the whole map. Pair errors are root-mean-square differences of aligned and
reference distances, and ten-neighbour recall compares stable-sorted neighbour lists. On
independent assays, the Poisson deviance splits exactly into total-sharing and allocation terms.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import logsumexp, xlogy

import msld

BANDS = [0, 15, 30, 60, 120, 240, 480, 960, 1920, np.inf]
SHELL_EDGES = [0, 10, 20, 40, 80, 160, 240, 383]
RANGES = [('ball', 0, k) for k in SHELL_EDGES[1:-1]] + [
    ('shell', a, b) for a, b in zip(SHELL_EDGES[:-1], SHELL_EDGES[1:])
]


def align(x, y):
    """x after one least-squares similarity onto y, reflection allowed."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, s, vt = np.linalg.svd(xc.T @ yc)
    return s.sum() / np.square(xc).sum() * xc @ (u @ vt) + y.mean(0)


def log_mean(z):
    """Fitted log means (S, n, n) of a saved map with coordinates, node_effects, intercepts and widths."""
    return msld.log_mean(z['coordinates'], z['node_effects'], z['intercepts'], z['widths'])


def recall10(x, y, idx):
    """Mean fraction of each returned recipient's ten nearest returned neighbours (map x of the
    recipients idx) that are among its ten nearest of all reference recipients y."""
    distance, truth = squareform(pdist(x)), squareform(pdist(y))
    np.fill_diagonal(distance, np.inf)
    np.fill_diagonal(truth, np.inf)
    predicted = idx[np.argsort(distance, axis=1, kind='stable')[:, : min(10, len(idx) - 1)]]
    actual = np.argsort(truth, axis=1, kind='stable')[:, :10]
    return float(np.mean([len(set(p) & set(actual[i])) / 10 for i, p in zip(idx, predicted)]))


def geometry_scores(x, y, idx, groups=None):
    """Pair RMSE after one similarity for the whole map, and ten-neighbour recall, of map x of the
    recipients idx. With field labels, the RMSE is split into within-field and cross-field pairs."""
    delta = pdist(align(x, y[idx])) - pdist(y[idx])
    out = {}
    if groups is None:
        out['pair_rmse'] = float(np.sqrt(np.mean(delta**2)))
    else:
        i, j = np.triu_indices(len(idx), 1)
        same = groups[idx][i] == groups[idx][j]
        out['within_pair_rmse'] = float(np.sqrt(np.mean(delta[same] ** 2)))
        out['between_pair_rmse'] = float(np.sqrt(np.mean(delta[~same] ** 2)))
    out['neighbor_recall10'] = recall10(x, y, idx)
    return out


def count_scores(eta, obs, mask):
    """Full, total-sharing and allocation deviance per observed evaluation count over the masked pairs.

    eta: fitted log means (S, pairs); obs: evaluation counts (S, pairs)."""
    total = obs.sum(0)
    lm = logsumexp(eta, axis=0)
    full = 2 * np.sum(xlogy(obs, obs) - obs * eta - obs + np.exp(eta), axis=0)
    pooled = 2 * (xlogy(total, total) - total * lm - total + np.exp(lm))
    allocation = 2 * np.sum(xlogy(obs, obs) - xlogy(obs, total[None]) - obs * (eta - lm[None]), axis=0)
    mass = float(total[mask].sum())
    return {
        name: float(v[mask].sum() / mass)
        for name, v in [('full', full), ('total', pooled), ('allocation', allocation)]
    }


def ranking(x):
    """Every other recipient of each recipient, nearest first, with stable tie breaking."""
    D = squareform(pdist(x))
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind='stable')[:, :-1]


def composition(x, y, labels):
    """Total variation between reconstructed and reference cell-type proportions in each neighbour-rank
    range (RANGES), averaged over recipients. A range beyond the last rank has TV None."""
    types, codes = np.unique(labels, return_inverse=True)
    onehot = np.eye(len(types))[codes]
    reference, reconstructed = ranking(y), ranking(x)
    rows = []
    for kind, a, b in RANGES:
        if a >= len(y) - 1:
            rows.append(dict(range=kind, lower=a, upper=b, composition_tv=None))
            continue
        tv = abs(onehot[reconstructed[:, a:b]].mean(1) - onehot[reference[:, a:b]].mean(1)).sum(1) / 2
        rows.append(dict(range=kind, lower=a, upper=b, composition_tv=float(tv.mean())))
    return rows


def composition_baseline(y, labels):
    """The same error when every recipient is given the leave-one-out composition of all recipients."""
    types, codes = np.unique(labels, return_inverse=True)
    onehot = np.eye(len(types))[codes]
    order = ranking(y)
    population = (onehot.sum(0)[None] - onehot) / (len(y) - 1)
    rows = []
    for kind, a, b in RANGES:
        if a >= len(y) - 1:
            rows.append(dict(range=kind, lower=a, upper=b, composition_tv=None))
            continue
        tv = abs(population - onehot[order[:, a:b]].mean(1)).sum(1) / 2
        rows.append(dict(range=kind, lower=a, upper=b, composition_tv=float(tv.mean())))
    return rows


def aligned_distances(z, y):
    """Pair distances of a map after one similarity onto the reference y."""
    return pdist(align(z['coordinates'], y))


def allocation(z, obs, ii, jj):
    """Allocation deviance of each pair (ii, jj) under a calibrated map."""
    d2 = pdist(z['coordinates'], 'sqeuclidean')
    eta = (
        z['intercepts'][:, None]
        + z['node_effects'][ii][None]
        + z['node_effects'][jj][None]
        - d2[None] / (2 * z['widths'][:, None])
    )
    logp = eta - logsumexp(eta, axis=0)[None]
    return 2 * np.sum(xlogy(obs, obs) - xlogy(obs, obs.sum(0)[None]) - obs * logp, axis=0)


def band_errors(pooled, labelled, y, mask):
    """Pair-distance RMSE of joint P and the pooled shape, both after the labelled head and one similarity,
    over all masked pairs (both bounds None) and in each reference-distance band (BANDS, in micrometres).
    The last band has no upper bound, and an empty band has RMSE None."""
    truth = pdist(y)
    errors = {
        name: (aligned_distances(z, y) - truth) ** 2
        for name, z in [('labelled', labelled), ('pooled', pooled)]
    }
    rows = []
    for lo, hi in [(None, None), *zip(BANDS[:-1], BANDS[1:])]:
        use = mask if lo is None else mask & (truth >= lo) & (truth < hi)
        row = dict(band_lower_um=lo, band_upper_um=hi if hi != np.inf else None, pairs=int(use.sum()))
        for name, e in errors.items():
            row[f'{name}_distance_rmse'] = float(np.sqrt(e[use].mean())) if use.any() else None
        rows.append(row)
    return rows


def allocation_gain_with_better_distance(pooled, labelled, obs, y, mask):
    """Positive allocation gains of joint P over the pooled shape, both after the labelled head; the part of
    them on pairs whose aligned distance is also closer to the reference; and the numbers of masked pairs
    with a closer distance and in all (ties within 1e-12 and 1e-9 excluded)."""
    ii, jj = np.triu_indices(len(y), 1)
    truth = pdist(y)
    gain = allocation(pooled, obs, ii, jj) - allocation(labelled, obs, ii, jj)
    geometric = (aligned_distances(pooled, y) - truth) ** 2 - (aligned_distances(labelled, y) - truth) ** 2
    closer = geometric > 1e-9
    positive = mask & (gain > 1e-12)
    both = positive & closer
    return float(gain[positive].sum()), float(gain[both].sum()), int((mask & closer).sum()), int(mask.sum())
