"""Check the saved example against the result tables at its saved parameters, without fitting anything.

For each of the four two-range acquisitions with separated fields, the script rebuilds the six starts
from the training counts; recomputes the objective of every fit of joint P and of the pooled competitor,
of the pooled continuations, of the nonspatial controls, of both calibration heads, and of the 23
calibrated comparison shapes; checks every selection; and recomputes the geometry, prediction, composition,
and distance-band endpoints, comparing them with the tables in results/. With --simulate, it also
regenerates the four assays of the first case from their seeds.
"""

import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

import msld
from controls import constant_log_mean, recipient_effects_log_mean, recipient_effects_objective
from endpoints import (
    allocation_gain_with_better_distance,
    band_errors,
    composition,
    composition_baseline,
    count_scores,
    geometry_scores,
    log_mean,
)
from graph import reconstruct
from head import head_objective
from reconstruct import profiled, starts
from simulate import IDENTITIES, PACKETS_PER_SITE, acquire, count_inputs, design_radii

# pandas 3 parses floats exactly by default; pandas 2 needs the round-trip parser.
EXACT = {} if int(pd.__version__.split('.')[0]) >= 3 else {'float_precision': 'round_trip'}
ANIMALS = ['animal1', 'animal2']
# Two technical replicates per animal.
CASES = [(22701, 'animal1'), (22702, 'animal1'), (22703, 'animal2'), (22704, 'animal2')]
COUNTS = dict(fits=0, heads=0, shapes=0, endpoint_values=0)
GAPS = [0.0]


def check(actual, expected, label, rel=1e-9, absolute=1e-9):
    actual, expected = np.nan if actual is None else float(actual), float(expected)
    if np.isnan(expected):
        assert np.isnan(actual), label
        return
    assert abs(actual - expected) <= absolute + rel * abs(expected), (label, actual, expected)
    GAPS.append(abs(actual - expected) / max(1.0, abs(expected)))


def load_json(path):
    return json.loads(Path(path).read_text())


def row(table, **match):
    selected = table[np.logical_and.reduce([table[k] == v for k, v in match.items()])]
    assert len(selected) == 1, match
    return selected.iloc[0]


def same_up_to_reflection(a, b, tol=1e-8):
    """Classical MDS fixes each axis only up to sign, so columns are compared up to sign."""
    scale = max(1.0, np.abs(b).max())
    return all(
        min(np.abs(a[:, k] - b[:, k]).max(), np.abs(a[:, k] + b[:, k]).max()) <= tol * scale
        for k in range(a.shape[1])
    )


def check_files():
    """Every example, data, and result file against the hashes in its manifest."""
    for manifest in ['examples/MANIFEST.json', 'data/MANIFEST.json']:
        for path, digest in load_json(ROOT / manifest)['files'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for entry in pd.read_csv(ROOT / 'results/MANIFEST.csv').itertuples():
        data = (ROOT / entry.file).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry.sha256_uncompressed, entry.file
        assert data.count(b'\n') == entry.rows + 1, entry.file


def check_fits(d, seed, train):
    """The six starts, every fit of joint P and of the pooled competitor, and the pooled continuations;
    returns the selected joint-P map, the final pooled map, and the pooled map before continuation."""
    C, widths = train['counts'], train['widths']
    n = C.shape[1]
    off = ~np.eye(n, dtype=bool)
    totals = C[:, off].sum(1)

    def labelled_value(z):
        theta = msld.pack(z['coordinates'], z['node_effects'], z['intercepts'])
        return msld.objective_and_gradient(theta, C, widths, n, off)[0]

    def pooled_value(z):
        theta = np.r_[z['coordinates'].ravel(), z['node_effects']]
        return profiled(theta, C.sum(0), widths, totals, n, off)[0]

    saved = np.load(d / 'inputs/starts.npz')
    rebuilt = starts(train, saved['G_coordinates'], saved['G_indices'], seed)
    assert all(same_up_to_reflection(r, s) for r, s in zip(rebuilt, saved['coordinates'])), 'starts'
    chosen = {}
    for method, value in [('labelled', labelled_value), ('pooled', pooled_value)]:
        selected = load_json(d / method / 'selected.json')
        for i, record in enumerate(selected['starts']):
            assert load_json(d / method / f'start{i}.json') == record
            check(value(np.load(d / method / f'start{i}.npz')), record['objective'], f'{method} start {i}')
            COUNTS['fits'] += 1
        assert selected['chosen'] == int(np.argmin([s['objective'] for s in selected['starts']]))
        chosen[method] = np.load(d / method / f"start{selected['chosen']}.npz")
    labelled, before = chosen['labelled'], chosen['pooled']
    continuation = load_json(d / 'pooled/continuation_starts.json')
    values = [pooled_value(np.load(d / f'labelled/start{i}.npz')) for i in range(6)]
    for v, recorded in zip(values, continuation['pooled_objective_at_labelled_starts']):
        check(v, recorded, 'pooled objective at a joint-P start')
    assert continuation['labelled_start'] == int(np.argmin(values))
    final = load_json(d / 'pooled/final.json')
    check(pooled_value(before), final['objective_before_continuation'], 'pooled before continuation')
    for k in range(2):
        value = pooled_value(np.load(d / f'pooled/continuation{k}.npz'))
        check(value, load_json(d / f'pooled/continuation{k}.json')['objective'], f'continuation {k}')
        check(value, final['continuation_objectives'][k], f'continuation {k} in the final record')
    pooled = np.load(d / 'pooled/final.npz')
    source = (
        before if final['chosen'] == 'before_continuation' else np.load(d / f"pooled/{final['chosen']}.npz")
    )
    assert all(np.array_equal(pooled[k], source[k]) for k in ['coordinates', 'node_effects', 'intercepts'])
    return labelled, pooled, before


def head_value(h, C, widths):
    """Objective of a calibration head at its saved parameters (total, not per count)."""
    off = ~np.eye(C.shape[1], dtype=bool)
    scaled = squareform(pdist(h['normalized_shape'], 'sqeuclidean'))[None] / (2 * widths[:, None, None])
    value, _ = head_objective(
        np.r_[h['node_effects'], h['intercepts'], h['fitted_squared_scale']], C, scaled, off
    )
    return value * C[:, off].sum()


def check_calibrations(d, train, shapes_of):
    """The nonspatial controls, both heads, and the 23 calibrated shapes with their selection; returns the
    log means of the controls, the heads, and the shapes by name."""
    C, widths = train['counts'], train['widths']
    n = C.shape[1]
    off = ~np.eye(n, dtype=bool)
    effects = np.load(d / 'controls/nonspatial_recipient_effects.npz')
    value, _ = recipient_effects_objective(
        np.r_[effects['node_effects_by_channel'].ravel(), effects['intercepts']], C
    )
    check(
        value * C[:, off].sum(),
        load_json(d / 'controls/nonspatial_recipient_effects.json')['objective'],
        'recipient effects',
    )
    shares = np.load(d / 'controls/nonspatial_constant.npz')
    eta = constant_log_mean(shares, n)
    check(
        np.sum((np.exp(eta) - C * eta)[:, off]),
        load_json(d / 'controls/nonspatial_constant.json')['objective'],
        'constant',
    )
    controls = dict(nonspatial_recipient_effects=recipient_effects_log_mean(effects), nonspatial_constant=eta)
    heads = {}
    for name, shape in shapes_of.items():
        h = np.load(d / f'heads/{name}.npz')
        x = shape['coordinates'] - shape['coordinates'].mean(0)
        assert np.allclose(
            x * np.sqrt(900 / pdist(x, 'sqeuclidean').mean()), h['normalized_shape'], rtol=0, atol=1e-9
        )
        assert np.allclose(h['coordinates'], h['normalized_shape'] * np.sqrt(h['fitted_squared_scale']))
        check(head_value(h, C, widths), load_json(d / f'heads/{name}.json')['objective'], f'{name} head')
        heads[name] = h
        COUNTS['heads'] += 1
    selection = load_json(d / 'shapes/selection.json')
    shapes = {}
    for status in selection['shapes']:
        z = np.load(d / f"shapes/{status['name']}.npz")
        idx = z['indices']
        check(head_value(z, C[:, idx[:, None], idx[None, :]], widths), status['objective'], status['name'])
        shapes[status['name']] = z
        COUNTS['shapes'] += 1
    chosen = {}
    for family in ['G', 'CMDS', 'UMAP', 'tSNE']:
        # UMAP and t-SNE members must cover every recipient; G keeps only the recipients it covers.
        members = [
            s
            for s in selection['shapes']
            if s['family'] == family and (family in ['G', 'CMDS'] or s['n_supported'] == n)
        ]
        chosen[family] = min(members, key=lambda s: s['objective'])['name']
    assert chosen == selection['selected']
    return controls, heads, shapes


def check_endpoints(seed, y, groups, labels, H, maps, heads, controls, tables):
    """Geometry, prediction, composition, and distance bands against the result tables."""
    n = len(y)
    ii, jj = np.triu_indices(n, 1)
    same = groups[ii] == groups[jj]
    obs = H[:, ii, jj]
    for method, z in maps.items():
        idx = z['indices'] if 'indices' in z else np.arange(n)
        expected = row(tables['geometry'], seed=seed, design='mix25', method=method)
        check(len(idx) / n, expected.coverage, f'{method} coverage')
        for key, value in geometry_scores(z['coordinates'], y, idx, groups).items():
            check(value, expected[key], f'{method} {key}')
            COUNTS['endpoint_values'] += 1
    scored = [
        (method, 'own_parameters', log_mean(maps[method]))
        for method in ['P', 'pooled', 'pooled_before_continuation']
    ]
    scored += [(method, 'own_parameters', eta) for method, eta in controls.items()]
    scored += [
        ('P', 'labelled_head', log_mean(heads['labelled'])),
        ('pooled', 'labelled_head', log_mean(heads['pooled'])),
    ]
    full_shapes = [name for name, z in maps.items() if 'indices' in z and len(z['indices']) == n]
    scored += [(name, 'labelled_head', log_mean(maps[name])) for name in full_shapes]
    for method, calibration, eta in scored:
        for scope, mask in [('all', np.ones(len(ii), bool)), ('within', same), ('between', ~same)]:
            expected = row(
                tables['prediction'],
                seed=seed,
                design='mix25',
                method=method,
                calibration=calibration,
                scope=scope,
            )
            for key, value in count_scores(eta[:, ii, jj], obs, mask).items():
                check(value, expected[key], f'{method} {calibration} {scope} {key}')
                COUNTS['endpoint_values'] += 1
    table = tables['composition']
    for method in table[(table.seed == seed) & (table.design == 'mix25')].method.unique():
        for r in composition(maps[method]['coordinates'], y, labels):
            expected = row(
                tables['composition'],
                seed=seed,
                design='mix25',
                method=method,
                kind=r['range'],
                rank_from=r['lower'],
                rank_to=r['upper'],
            )
            check(r['composition_tv'], expected.composition_tv, f'{method} composition')
            COUNTS['endpoint_values'] += 1
    bands = tables['bands'][tables['bands'].seed == seed]
    rows = band_errors(heads['pooled'], heads['labelled'], y, same)
    assert len(rows) == len(bands)
    for r in rows:
        lower = (
            bands.band_lower_um.isna()
            if r['band_lower_um'] is None
            else bands.band_lower_um == r['band_lower_um']
        )
        expected = bands[lower].iloc[0]
        assert expected.pairs == r['pairs']
        for key in ['labelled_distance_rmse', 'pooled_distance_rmse']:
            check(r[key], expected[key], f'band {key}')
            COUNTS['endpoint_values'] += 1
    return allocation_gain_with_better_distance(heads['pooled'], heads['labelled'], obs, y, same)


def regenerate(seed, animal):
    """Regenerate the four assays of one case; packet records and counts must match exactly."""
    d = ROOT / 'examples' / f'seed{seed}_{animal}_mix25'
    g = np.load(ROOT / 'data/fields' / f'{animal}_geometry.npz')
    incidence, records = acquire(g['coordinates'], g['target'], seed, 'mix25')
    for rep, record in enumerate(records):
        saved = np.load(d / f'inputs/packets_rep{rep}.npz')
        assert all(np.array_equal(np.asarray(record[key]), saved[key]) for key in saved.files), rep
    radii = design_radii('mix25')[0]
    for split, inputs in count_inputs(incidence, radii).items():
        saved = np.load(d / f'inputs/{split}_input.npz')
        assert all(np.array_equal(inputs[key], saved[key]) for key in saved.files), split
    Y = {float(r): sparse.hstack(incidence[r][:2], format='csr') for r in radii}
    gx, idx, _ = reconstruct(Y, 2 * PACKETS_PER_SITE * len(g['coordinates']) * IDENTITIES, len(g['target']))
    saved = np.load(d / 'inputs/starts.npz')
    assert np.array_equal(idx, saved['G_indices']) and same_up_to_reflection(gx, saved['G_coordinates'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--simulate',
        action='store_true',
        help='also regenerate the assays of the first case',
    )
    args = ap.parse_args()
    check_files()

    def read(name):
        return pd.read_csv(ROOT / 'results/fields' / name, **EXACT)

    tables = dict(
        geometry=read('geometry.csv'),
        prediction=read('prediction.csv'),
        composition=read('composition.csv'),
        bands=read('distance_bands.csv'),
    )
    gains = {animal: np.zeros(4) for animal in ANIMALS}
    for seed, animal in CASES:
        d = ROOT / 'examples' / f'seed{seed}_{animal}_mix25'
        g = np.load(ROOT / 'data/fields' / f'{animal}_geometry.npz')
        train = np.load(d / 'inputs/train_input.npz')
        labelled, pooled, before = check_fits(d, seed, train)
        controls, heads, shapes = check_calibrations(d, train, dict(labelled=labelled, pooled=pooled))
        maps = dict(P=labelled, pooled=pooled, pooled_before_continuation=before, **shapes)
        H = np.load(d / 'inputs/evaluation_input.npz')['counts']
        gains[animal] += check_endpoints(
            seed, g['coordinates'][g['target']], g['groups'], g['labels'], H, maps, heads, controls, tables
        )
    baseline = read('composition_without_position.csv')
    for animal in ANIMALS:
        g = np.load(ROOT / 'data/fields' / f'{animal}_geometry.npz')
        for r in composition_baseline(g['coordinates'][g['target']], g['labels']):
            expected = row(baseline, animal=animal, kind=r['range'], rank_from=r['lower'], rank_to=r['upper'])
            check(r['composition_tv'], expected.composition_tv, 'composition without position')
            COUNTS['endpoint_values'] += 1
    if args.simulate:
        regenerate(*CASES[0])
    share = {animal: round(100 * g[1] / g[0], 2) for animal, g in gains.items()}
    closer = {animal: round(100 * g[2] / g[3], 2) for animal, g in gains.items()}
    print(
        json.dumps(
            dict(
                complete=True,
                cases=len(CASES),
                **COUNTS,
                max_relative_gap=max(GAPS),
                allocation_gain_on_better_placed_pairs_pct=share,
                within_field_pairs_placed_closer_pct=closer,
                simulated=args.simulate,
            )
        )
    )


if __name__ == '__main__':
    main()
