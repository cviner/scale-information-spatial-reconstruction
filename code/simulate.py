"""Simulated molecular transport and capture on measured cell positions.

For each of 6N packets: an anchor drawn uniformly from all N measured cells; a loading
L ~ LogNormal(0, 0.5^2) and m = max(1, round(40 L)) identity slots, cycled over 40 identities
unique to the packet; one radius drawn from the design's list ([r], or [15, 480, 480, 480] um for
mix25); and 12 copies of each slot, all at the packet's radius. Each copy is kept with probability 0.8,
displaced from the anchor by N(0, r^2 I) and captured by the nearest measured cell. Incidence is
binary per channel. All measured cells compete for capture; only the observed recipients'
rows are kept for inference.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

RADII = [15, 30, 60, 90, 120, 150, 180, 210, 240, 300, 360, 480]  # micrometres
DESIGNS = [f'r{r}' for r in RADII] + ['mix25']
PACKETS_PER_SITE, IDENTITIES, COPIES, CAPTURE, LOADING_SD = 6, 40, 12, 0.8, 0.5


class IndexedCapture:
    """Nearest measured cell by k-d tree; exact distance ties are resolved by brute force."""

    def __init__(self):
        self.coords = None
        self.tree = None

    def __call__(self, coords, pos):
        if self.coords is not coords:
            self.coords = coords
            self.tree = cKDTree(coords)
        dist, idx = self.tree.query(pos, k=2, workers=1)
        result = idx[:, 0].astype(np.int32)
        for j in np.flatnonzero(abs(dist[:, 0] - dist[:, 1]) < 1e-10):
            result[j] = np.argmin(np.square(coords - pos[j]).sum(1))
        return result


def simulate_assay(coords, rng, radius_list, copies_per_entry, capture_nearest):
    """One assay, with one radius per packet drawn from radius_list. Returns ({radius: binary
    cell-by-identity incidence (CSR)}, the anchor, radius choice, loaded identity slots and captured
    copies of every packet)."""
    n_cells = len(coords)
    n_packets = PACKETS_PER_SITE * n_cells
    rows = {r: [] for r in radius_list}
    cols = {r: [] for r in radius_list}
    record = dict(anchor=[], choice=[], loaded_identities=[], captured_molecules=[])
    # Each identity slot has copies_per_entry copies for each entry of the radius list.
    total_copies = copies_per_entry * len(radius_list)
    for p in range(n_packets):
        anchor = rng.randint(0, n_cells)
        k = max(1, int(round(IDENTITIES * rng.lognormal(0.0, LOADING_SD))))
        choice = rng.randint(0, len(radius_list))
        r = radius_list[choice]
        identity = np.repeat(p * IDENTITIES + np.arange(k) % IDENTITIES, total_copies)
        identity = identity[rng.rand(identity.size) < CAPTURE]
        record['anchor'].append(anchor)
        record['choice'].append(choice)
        record['loaded_identities'].append(k)
        record['captured_molecules'].append(identity.size)
        if identity.size:
            rows[r].append(capture_nearest(coords, coords[anchor][None, :] + rng.randn(identity.size, 2) * r))
            cols[r].append(identity)
    X = {}
    for r in radius_list:
        ri = np.concatenate(rows[r]) if rows[r] else np.empty(0, np.int32)
        ci = np.concatenate(cols[r]) if cols[r] else np.empty(0, np.int64)
        M = sparse.coo_matrix((np.ones(ri.size), (ri, ci)), shape=(n_cells, n_packets * IDENTITIES)).tocsr()
        M.sum_duplicates()
        M.data[:] = 1.0
        X[r] = M
    return X, {k: np.asarray(v) for k, v in record.items()}


def design_radii(design):
    """(channel radii, radius list sampled per packet, copies per list entry) of a design."""
    if design == 'mix25':
        return [15, 480], [15, 480, 480, 480], 3
    return [int(design[1:])], [int(design[1:])], COPIES


def acquire(coords, target, seed, design, assays=4):
    """Four independent assays of one design.

    Returns ({radius: [recipient incidence per assay]}, packet records per assay).
    """
    radii, choices, copies = design_radii(design)
    capture = IndexedCapture()
    incidence = {r: [] for r in radii}
    records = []
    for rep in range(assays):
        Y, record = simulate_assay(coords, np.random.RandomState(seed * 100 + rep), choices, copies, capture)
        record.update(
            nominal_molecules=PACKETS_PER_SITE * len(coords) * IDENTITIES * COPIES,
            realized_molecules=int(record['loaded_identities'].sum() * COPIES),
        )
        records.append(record)
        for r in radii:
            incidence[r].append(Y[r][target].tocsr())
    return incidence, records


def count_inputs(incidence, radii):
    """Training (assays 0, 1) and evaluation (assays 2, 3) sharing counts: the per-channel Gram
    matrices averaged over two assays, their diagonal loads, the pooled cosine similarity, and the
    widths sigma_s^2 = 2 r_s^2."""
    widths = 2 * np.square(radii).astype(float)
    out = {}
    for split, reps in [('train', [0, 1]), ('evaluation', [2, 3])]:
        C = np.array([sum((incidence[r][k] @ incidence[r][k].T).toarray() for k in reps) / 2 for r in radii])
        q = np.diagonal(C, axis1=1, axis2=2).copy()
        load = q.sum(0)
        cos = np.clip(C.sum(0) / np.sqrt(np.maximum(load[:, None] * load[None], 1)), 0, 1)
        np.fill_diagonal(cos, 1)
        for m in C:
            np.fill_diagonal(m, 0)
        out[split] = dict(counts=C, row_load=q, cosine=cos, widths=widths)
    return out
