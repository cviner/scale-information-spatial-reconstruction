"""Tests of the recipient geometry, rebuilt from the included sections, and of the whole pipeline on a small
synthetic layout."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from controls import constant, recipient_effects
from endpoints import count_scores, geometry_scores, log_mean
from geometry import build_geometry
from graph import reconstruct as graph_g
from head import calibrate
from reconstruct import continue_pooled, fit, starts
from simulate import IDENTITIES, PACKETS_PER_SITE, acquire, count_inputs, design_radii


def test_geometry_rebuilds_from_sections():
    for animal in ['animal1', 'animal2']:
        section = pd.read_csv(ROOT / 'data/sections' / f'{animal}_section.csv.gz', dtype={'Cell_ID': str})
        for design in ['fields', 'distributed']:
            mine = build_geometry(section, design)
            saved = np.load(ROOT / 'data' / design / f'{animal}_geometry.npz')
            for key in ['ids', 'target', 'groups', 'labels', 'class_labels']:
                assert np.array_equal(mine[key], saved[key]), (animal, design, key)
            assert np.allclose(mine['coordinates'], saved['coordinates'], rtol=0, atol=1e-9)


def test_toy_pipeline_runs_end_to_end():
    rng = np.random.default_rng(0)
    coords = np.vstack([rng.uniform(0, 300, size=(150, 2)), rng.uniform(900, 1200, size=(150, 2))])
    target = np.r_[0:20, 150:170]
    incidence, records = acquire(coords, target, 101, 'mix25')
    assert len(records) == 4 and all(len(r['anchor']) == PACKETS_PER_SITE * len(coords) for r in records)
    radii = design_radii('mix25')[0]
    inputs = count_inputs(incidence, radii)
    train, evaluation = inputs['train'], inputs['evaluation']
    assert np.array_equal(train['counts'], train['counts'].transpose(0, 2, 1))
    Y = {float(r): sparse.hstack(incidence[r][:2], format='csr') for r in radii}
    gx, idx, geo = graph_g(Y, 2 * PACKETS_PER_SITE * len(coords) * IDENTITIES, len(target))
    x0 = starts(train, gx, idx, 101)
    assert x0.shape == (6, len(target), 2)
    states, records, chosen = fit(train['counts'], train['widths'], x0[:2], 'labelled')
    x = states[chosen][0]
    pstates, _, pchosen = fit(train['counts'], train['widths'], x0[:2], 'pooled')
    _, crecords, cchosen, _ = continue_pooled(train['counts'], train['widths'], pstates[pchosen], states)
    assert len(crecords) == 2 and cchosen in ('before_continuation', 0, 1)
    for control in (recipient_effects, constant):
        _, record = control(train['counts'])
        assert np.isfinite(record['objective'])
    head, _ = calibrate(x, train['counts'], train['widths'])
    y = coords[target]
    groups = np.repeat([0, 1], 20)
    ii, jj = np.triu_indices(len(target), 1)
    scores = geometry_scores(x, y, np.arange(len(target)), groups)
    assert 0 <= scores['neighbor_recall10'] <= 1 and scores['between_pair_rmse'] >= 0
    assert 'pair_rmse' in geometry_scores(x, y, np.arange(len(target)))
    result = count_scores(log_mean(head)[:, ii, jj], evaluation['counts'][:, ii, jj], np.ones(len(ii), bool))
    assert abs(result['full'] - result['total'] - result['allocation']) < 1e-9


def test_section_selection_on_a_synthetic_table(tmp_path):
    from geometry import extract_section, select_sections

    rows = []
    for animal, behavior in [(3, 'Naive'), (1, 'Naive'), (2, 'Naive'), (0, 'Virgin Parenting')]:
        for bregma in [-0.24, 0.01, 0.06]:
            for k in range(3):
                rows.append(
                    dict(
                        Cell_ID=f'{animal}-{bregma}-{k}',
                        Animal_ID=animal,
                        Animal_sex='Female',
                        Behavior=behavior,
                        Bregma=bregma,
                        Centroid_X=float(k),
                        Centroid_Y=float(animal),
                        Cell_class='Inhibitory',
                        Neuron_cluster_ID='I-1',
                        Gene_A=1.0,
                    )
                )
    duplicate = next(dict(r) for r in rows if r['Cell_ID'] == '1-0.01-1')
    duplicate['Cell_ID'] = '1-0.01-9'  # A second record at the same centroid, with identical values.
    rows.append(duplicate)
    path = tmp_path / 'cells.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    selections = select_sections(path)
    assert [s['animal'] for s in selections] == [1, 2] and all(s['bregma'] == 0.01 for s in selections)
    assert '1-0.01-9' in selections[0]['cell_ids']
    section = extract_section(path, selections[0]['cell_ids'])
    assert list(section.Cell_ID) == ['1-0.01-0', '1-0.01-1', '1-0.01-2']


def test_run_case_writes_the_layout_of_the_examples(tmp_path):
    import json
    import subprocess

    rng = np.random.default_rng(0)
    coords = np.vstack([rng.uniform(0, 300, size=(150, 2)), rng.uniform(900, 1200, size=(150, 2))])
    np.savez(
        tmp_path / 'toy.npz',
        coordinates=coords,
        target=np.r_[0:20, 150:170],
        groups=np.repeat([0, 1], 20),
        labels=np.array(['A', 'B', 'C', 'D'])[rng.integers(4, size=40)],
    )
    out = tmp_path / 'case'
    subprocess.run(
        [
            sys.executable,
            str(ROOT / 'run_case.py'),
            '--geometry',
            str(tmp_path / 'toy.npz'),
            '--seed',
            '7',
            '--design',
            'mix25',
            '--out',
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    example = ROOT / 'examples' / 'seed22701_animal1_mix25'

    def layout(root):
        return {
            str(p.relative_to(root))
            for p in root.rglob('*')
            if p.is_file() and p.parts[len(root.parts)] != 'shapes'
        }

    assert layout(out) - {'scores.json'} == layout(example)
    scores = json.loads((out / 'scores.json').read_text())
    # All within-field pairs and the nine bands.
    assert {'P', 'pooled', 'P_head', 'pooled_head'} <= set(scores) and len(scores['distance_bands']) == 10
    for name in sorted(layout(example)):
        mine, theirs = out / name, example / name
        if name.endswith('.npz'):
            assert set(np.load(mine).files) == set(np.load(theirs).files), name
        elif not name.startswith('heads/'):
            a, b = json.loads(mine.read_text()), json.loads(theirs.read_text())
            if isinstance(b, dict):
                assert set(b) <= set(a), name
