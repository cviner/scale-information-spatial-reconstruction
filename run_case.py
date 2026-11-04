"""Run one acquisition case: simulate four assays, fit joint P (and, for a two-range acquisition, the
pooled competitor with its continuations and the nonspatial controls), calibrate the heads, optionally
calibrate the 23 comparison shapes, and score everything.

    python run_case.py --geometry data/fields/animal1_geometry.npz --seed 22701 --design mix25 --out work/case

The output folder has the layout of the cases in examples/.
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
import json

import numpy as np
from scipy import sparse

from controls import constant, constant_log_mean, recipient_effects, recipient_effects_log_mean
from endpoints import band_errors, composition, count_scores, geometry_scores, log_mean
from graph import reconstruct as reconstruct_g
from head import calibrate
from reconstruct import continue_pooled, fit, starts
from simulate import DESIGNS, IDENTITIES, PACKETS_PER_SITE, acquire, count_inputs, design_radii


def save(path, arrays, record=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if record is not None:
        write_json(path.with_suffix('.json'), record)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def state(x, a, b, widths):
    return dict(coordinates=x, node_effects=a, intercepts=b, widths=widths)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--geometry', type=Path, required=True, help='data/fields/... or data/distributed/...')
    ap.add_argument('--seed', type=int, required=True, help='seed of the technical replicate (22701 to 22704 in the paper)')
    ap.add_argument(
        '--design',
        choices=DESIGNS,
        required=True,
        help='r15 ... r480: one radius in um; mix25: 15 um with probability 1/4, else 480 um',
    )
    ap.add_argument(
        '--shapes', action='store_true', help='also calibrate the 23 comparison shapes (needs umap-learn)'
    )
    ap.add_argument('--out', type=Path, required=True, help='new output folder')
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError('The output exists; choose a new path')
    out = args.out
    g = np.load(args.geometry)
    coords, target = g['coordinates'], g['target']
    n = len(target)
    # Assays 0 and 1 train the fits, and assays 2 and 3 evaluate them.
    incidence, records = acquire(coords, target, args.seed, args.design)
    for rep, r in enumerate(records):
        save(
            out / 'inputs' / f'packets_rep{rep}.npz',
            {k: r[k] for k in ['anchor', 'choice', 'loaded_identities', 'captured_molecules']},
            dict(
                rep=rep,
                nominal_molecules=r['nominal_molecules'],
                realized_molecules=r['realized_molecules'],
                captured_molecules=int(np.sum(r['captured_molecules'])),
            ),
        )
    radii = design_radii(args.design)[0]
    inputs = count_inputs(incidence, radii)
    for split, arrays in inputs.items():
        save(out / 'inputs' / f'{split}_input.npz', arrays)
    train = inputs['train']
    C, widths = train['counts'], train['widths']
    Y = {float(r): sparse.hstack(incidence[r][:2], format='csr') for r in radii}
    gx, g_indices, geodesic = reconstruct_g(Y, 2 * PACKETS_PER_SITE * len(coords) * IDENTITIES, n)
    x0 = starts(train, gx, g_indices, args.seed)
    save(out / 'inputs' / 'starts.npz', dict(coordinates=x0, G_coordinates=gx, G_indices=g_indices))
    maps, support, fits = {}, {}, {}
    for method in ['labelled', 'pooled'] if len(radii) > 1 else ['labelled']:
        states, starts_records, chosen = fit(C, widths, x0, method)
        for i, ((x, a, b), r) in enumerate(zip(states, starts_records)):
            save(out / method / f'start{i}.npz', state(x, a, b, widths), r)
        write_json(out / method / 'selected.json', dict(chosen=chosen, starts=starts_records))
        fits[method] = (states, chosen)
    states, chosen = fits['labelled']
    maps['P'] = state(*states[chosen], widths)
    if len(radii) > 1:
        pstates, pchosen = fits['pooled']
        final, crecords, cchosen, details = continue_pooled(C, widths, pstates[pchosen], states)
        for k, ((x, a, b), r) in enumerate(zip(details['states'], crecords)):
            save(out / 'pooled' / f'continuation{k}.npz', state(x, a, b, widths), r)
        write_json(
            out / 'pooled' / 'continuation_starts.json',
            dict(
                labelled_start=details['labelled_start'],
                pooled_objective_at_labelled_starts=details['pooled_objective_at_labelled_starts'],
            ),
        )
        save(out / 'pooled' / 'final.npz', state(*final, widths))
        write_json(
            out / 'pooled' / 'final.json',
            dict(
                chosen=cchosen if cchosen == 'before_continuation' else f'continuation{cchosen}',
                objective_before_continuation=details['objective_before_continuation'],
                continuation_objectives=[r['objective'] for r in crecords],
            ),
        )
        maps['pooled'] = state(*final, widths)
        effects, record = recipient_effects(C)
        save(out / 'controls' / 'nonspatial_recipient_effects.npz', effects, record)
        maps['nonspatial_recipient_effects'] = dict(log_mean=recipient_effects_log_mean(effects))
        shares, record = constant(C)
        save(out / 'controls' / 'nonspatial_constant.npz', shares, record)
        maps['nonspatial_constant'] = dict(log_mean=constant_log_mean(shares, n))
    for name, head_name in [('P', 'labelled'), ('pooled', 'pooled')]:
        if name in maps:
            head, record = calibrate(maps[name]['coordinates'], C, widths)
            save(out / 'heads' / f'{head_name}.npz', head, record)
            maps[f'{name}_head'] = head
    if args.shapes:
        from embeddings import candidates

        shapes, statuses, selected = candidates(train, gx, g_indices, geodesic)
        for name, (head, indices) in shapes.items():
            save(out / 'shapes' / f'{name}.npz', dict(indices=indices, **head))
        write_json(out / 'shapes' / 'selection.json', dict(selected=selected, shapes=statuses))
        for name in selected.values():
            maps[name], support[name] = shapes[name]
    y, groups, labels = coords[target], g['groups'], g['labels']
    H = np.load(out / 'inputs' / 'evaluation_input.npz')['counts']
    ii, jj = np.triu_indices(n, 1)
    fields = len(np.unique(groups)) > 1
    same = groups[ii] == groups[jj]
    scopes = {'all': np.ones(len(ii), bool)}
    if fields:
        scopes.update(within=same, between=~same)
    result = {}
    for name, z in maps.items():
        idx = support.get(name, np.arange(n))
        entry = dict(coverage=len(idx) / n)
        if 'coordinates' in z:
            entry['geometry'] = geometry_scores(z['coordinates'], y, idx, groups if fields else None)
        if len(idx) == n:
            eta = z['log_mean'] if 'log_mean' in z else log_mean(z)
            entry['prediction'] = {
                scope: count_scores(eta[:, ii, jj], H[:, ii, jj], mask) for scope, mask in scopes.items()
            }
            if 'coordinates' in z:
                entry['composition'] = composition(z['coordinates'], y, labels)
        result[name] = entry
    if 'pooled_head' in maps:
        result['distance_bands'] = band_errors(
            maps['pooled_head'], maps['P_head'], y, same if fields else scopes['all']
        )
    write_json(out / 'scores.json', result)
    print(out / 'scores.json')


if __name__ == '__main__':
    main()
