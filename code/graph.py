"""Graph reconstruction G: a chance-overlap correction, a Bonferroni threshold, precision-weighted
fusion of the channel distances, geodesic completion on the largest component, and classical MDS.

For channel incidence X with raw overlap A = X X^T, loads q (the diagonal of A) and library size L,
the chance overlap is E = q q^T / L, the score is z = (A - E) / sqrt(E), the chance-corrected
Ochiai similarity is s = (A - E) / sqrt(q q^T), and the distance is d = 2 r sqrt(-log s). An edge
enters if z >= Phi^-1(1 - 0.05 / n_pairs) and d is positive and finite; channels are fused with
weights z / r^2. The weights are heuristic, and the cut does not control errors exactly for these
dependent counts.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import norm


def channel_scores(X, r, n_identities, s_floor=1e-6):
    """Scores z and distances d (n x n, symmetric, zero diagonal) from one channel's incidence X."""
    X = X.astype(np.float64)
    A = np.asarray((X @ X.T).todense())
    q = np.diag(A).copy()
    E = np.outer(q, q) / float(n_identities)
    Z = (A - E) / np.sqrt(np.clip(E, 1e-12, None))
    np.fill_diagonal(Z, 0.0)
    root = np.sqrt(np.clip(q, 1e-12, None))
    S = (A - E) / np.outer(root, root)
    D = 2.0 * r * np.sqrt(-np.log(np.clip(S, s_floor, 1.0)))
    np.fill_diagonal(D, 0.0)
    # Symmetrize against floating-point asymmetry.
    return 0.5 * (Z + Z.T), 0.5 * (D + D.T)


def bonferroni_z(n, alpha=0.05):
    """z* = Phi^-1(1 - alpha / n_pairs), with n_pairs = n (n - 1) / 2."""
    return float(norm.ppf(1.0 - alpha / (n * (n - 1) / 2.0)))


def fuse_edges(channels, z_thresh):
    """Fused distances and the mask of accepted edges, from (r, Z, D) per channel.

    An edge of a channel enters if z >= z_thresh and its distance is positive and finite, with
    weight z / r^2: its reliability times the resolution of the channel.
    """
    n = len(channels[0][1])
    num, den = np.zeros((n, n)), np.zeros((n, n))
    for r, Z, D in channels:
        observed = (Z >= z_thresh) & np.isfinite(D) & (D > 0)
        np.fill_diagonal(observed, False)
        w = np.where(observed, Z / (r**2), 0.0)
        num += w * np.where(observed, D, 0.0)
        den += w
    mask = den > 0
    fused = np.zeros((n, n))
    fused[mask] = num[mask] / den[mask]
    fused = 0.5 * (fused + fused.T)
    mask = mask & mask.T
    np.fill_diagonal(mask, False)
    np.fill_diagonal(fused, 0.0)
    return fused, mask


def classical_mds(D, n_dim=2):
    """Classical (Torgerson) MDS of a full distance matrix."""
    D2 = D**2
    # Double centring without forming J = I - 11'/n.
    B = -0.5 * (D2 - D2.mean(axis=1, keepdims=True) - D2.mean(axis=0, keepdims=True) + float(D2.mean()))
    B = 0.5 * (B + B.T)
    vals, vecs = eigh(B, subset_by_index=[len(D) - n_dim, len(D) - 1])
    order = np.argsort(vals)[::-1]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def reconstruct(incidence, n_identities, n):
    """G from the training incidence {radius: recipients x identities}.

    Returns (coordinates, indices, geodesic distances); the coordinates cover only the largest
    connected component, whose recipients are the indices.
    """
    channels = [(r, *channel_scores(incidence[r], r, n_identities)) for r in sorted(incidence)]
    edges, accepted = fuse_edges(channels, bonferroni_z(n))
    adj = sparse.csr_matrix(np.where(accepted, edges, 0))
    _, lab = connected_components(adj, directed=False)
    idx = np.flatnonzero(lab == np.argmax(np.bincount(lab)))
    geo = shortest_path(adj[idx][:, idx], directed=False, method='D')
    gx = classical_mds(geo) if len(idx) > 2 else np.zeros((len(idx), 2))
    return gx, idx, geo
