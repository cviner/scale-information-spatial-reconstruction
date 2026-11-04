"""The 23 comparison shapes per acquisition, each given the calibration head.

G, classical MDS of sqrt(-2 r_max^2 log c) on the pooled cosine similarity c, twelve UMAP maps on
the cosine distance (10 or 25 neighbours, minimum distance 0.1 or 0.99, seeds 0-2), three UMAP maps
on G's geodesic distances (25 neighbours, minimum distance 0.99) and six exact t-SNE maps on the
cosine distance (perplexity 10 or 30). Within each family, the member with the lowest training
objective among maps covering every recipient is selected; G keeps only the recipients it covers.
"""

import warnings

import numpy as np

from graph import classical_mds
from head import calibrate


def configurations():
    """The 23 comparison shapes as (name, family, settings)."""
    rows = [('G', 'G', None), ('CMDS', 'CMDS', None)]
    rows += [
        (f'UMAP_cosine_k{k}_md{md}_s{s}', 'UMAP', ('cosine', k, md, s))
        for k in [10, 25]
        for md in [0.1, 0.99]
        for s in [0, 1, 2]
    ]
    rows += [(f'UMAP_geodesic_k25_md0.99_s{s}', 'UMAP', ('geodesic', 25, 0.99, s)) for s in [0, 1, 2]]
    rows += [(f'tSNE_p{p}_s{s}', 'tSNE', (p, s)) for p in [10, 30] for s in [0, 1, 2]]
    return rows


def shape(name, family, config, train, gx, g_indices, geodesic):
    """Uncalibrated coordinates and the recipients they cover."""
    n = len(train['cosine'])
    indices = np.arange(n)
    cosine = np.maximum(1 - train['cosine'], 0)
    np.fill_diagonal(cosine, 0)
    if name == 'G':
        return gx.copy(), g_indices
    if name == 'CMDS':
        return classical_mds(
            np.sqrt(-max(train['widths']) * np.log(np.clip(train['cosine'], 1e-6, 1)))
        ), indices
    if family == 'UMAP':
        import umap

        metric, k, md, seed = config
        d = cosine if metric == 'cosine' else geodesic
        if metric == 'geodesic':
            indices = g_indices
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = umap.UMAP(
                metric='precomputed',
                n_components=2,
                n_neighbors=min(k, len(d) - 1),
                min_dist=md,
                n_epochs=1000,
                random_state=seed,
                n_jobs=1,
                low_memory=True,
            ).fit_transform(d)
        return x, indices
    from sklearn.manifold import TSNE

    perplexity, seed = config
    x = TSNE(
        metric='precomputed',
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init='random',
        method='exact',
        learning_rate='auto',
        max_iter=1000,
    ).fit_transform(cosine)
    return x, indices


def candidates(train, gx, g_indices, geodesic):
    """Calibrate all 23 shapes; returns ({name: (fit, indices)}, statuses, selected names by family)."""
    counts, widths = train['counts'], train['widths']
    maps, statuses = {}, []
    for name, family, config in configurations():
        x, indices = shape(name, family, config, train, gx, g_indices, geodesic)
        fit, status = calibrate(x, counts[:, indices[:, None], indices[None, :]], widths)
        maps[name] = (fit, indices)
        statuses.append(dict(status, name=name, family=family, n_supported=len(indices)))
    n = len(train['cosine'])
    selected = {}
    for family in ['G', 'CMDS', 'UMAP', 'tSNE']:
        members = [s for s in statuses if s['family'] == family]
        if family in ['UMAP', 'tSNE']:
            members = [s for s in members if s['n_supported'] == n]
        selected[family] = min(members, key=lambda s: s['objective'])['name']
    return maps, statuses, selected
