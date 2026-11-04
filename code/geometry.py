"""Measured tissue geometry: sections, separated fields and distributed recipients.

Source: the MERFISH cell table of Moffitt, Bambah-Mukku et al. (Science 2018), Dryad
doi:10.5061/dryad.8t8s248, file Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv.
The selection takes the two lowest numeric animal identifiers marked naive and, for each, the
section with bregma closest to zero. Every finite centroid is a capture site, and records that share
a centroid are collapsed to one.
"""

import hashlib
import unicodedata

import numpy as np
import pandas as pd

SOURCE_BYTES = 1031419408
SOURCE_MD5 = '25a51abdf981039949cfdaf4db0a9ab3'
# Salts of the two hash orders (Supplementary Methods S1).
FIELD_SALT = 'R219:'
DISTRIBUTED_SALT = 'R229:'
META = [
    'Cell_ID',
    'Animal_ID',
    'Animal_sex',
    'Behavior',
    'Bregma',
    'Centroid_X',
    'Centroid_Y',
    'Cell_class',
    'Neuron_cluster_ID',
]


def verify_source(path):
    """Check the size and MD5 of the downloaded cell table; return its SHA-256."""
    if path.stat().st_size != SOURCE_BYTES:
        raise ValueError(f'{path}: expected {SOURCE_BYTES} bytes, the size of the Dryad cell table')
    md5 = hashlib.md5()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            md5.update(chunk)
            h.update(chunk)
    if md5.hexdigest() != SOURCE_MD5:
        raise ValueError(f'{path}: MD5 differs from the Dryad cell table')
    return h.hexdigest()


def select_sections(path):
    """The two lowest naive animal identifiers and, for each, the section closest to bregma 0."""
    data = pd.read_csv(path, usecols=META, dtype={'Cell_ID': str})
    if data.Cell_ID.duplicated().any():
        raise ValueError('duplicate cell identifiers')

    def normalize(value):
        return unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().strip().lower()

    naive = data.Behavior.map(normalize) == 'naive'
    animals = sorted(pd.to_numeric(data.loc[naive, 'Animal_ID']).unique())[:2]
    if len(animals) != 2 or not all(int(v) == v for v in animals):
        raise ValueError('expected at least two integer naive animal identifiers')
    selections = []
    for animal in map(int, animals):
        subset = data[(data.Animal_ID == animal) & naive]
        bregma = float(sorted(subset.Bregma.dropna().unique(), key=lambda x: (abs(x), x))[0])
        chosen = subset[subset.Bregma == bregma]
        if not np.isfinite(chosen[['Centroid_X', 'Centroid_Y']].to_numpy()).all():
            raise ValueError(f'nonfinite centroids in the section of animal {animal}')
        selections.append(dict(animal=animal, bregma=bregma, cell_ids=chosen.Cell_ID.tolist()))
    return selections


def extract_section(path, cell_ids):
    """All columns for the selected cells, sorted by identifier, with shared centroids collapsed."""
    wanted = set(cell_ids)
    chunks = []
    for block in pd.read_csv(path, dtype={'Cell_ID': str}, chunksize=50000):
        retained = block[block.Cell_ID.isin(wanted)]
        if len(retained):
            chunks.append(retained)
    sub = pd.concat(chunks, ignore_index=True).sort_values('Cell_ID', kind='stable')
    remove = []
    for _, part in sub.groupby(['Centroid_X', 'Centroid_Y']):
        if len(part) < 2:
            continue
        values = part.drop(columns=['Cell_ID'])
        if not all(values.iloc[0].equals(values.iloc[j]) for j in range(1, len(values))):
            raise ValueError('distinct records at identical centroids')
        remove.extend(part.index[1:])
    return sub.drop(index=remove).reset_index(drop=True)


def annotations(section):
    """Neuronal cluster where available, otherwise major cell class."""
    classes = section.Cell_class.fillna('Missing').astype(str)
    clusters = section.Neuron_cluster_ID
    return np.array(
        [f'Neuron_{c}' if pd.notna(c) else str(k) for k, c in zip(classes, clusters)]
    ), classes.to_numpy(str)


def hash_order(ids, salt):
    """Indices in ascending order of SHA-256(salt + identifier)."""
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256((salt + ids[i]).encode()).hexdigest())


def separated_fields(xy, ids, fields=4, size=96, spacing=500.0):
    """Four fields of the 96 cells nearest centres visited in hash order and at least 500 um apart."""
    centers = []
    for i in hash_order(ids, FIELD_SALT):
        if all(np.linalg.norm(xy[i] - xy[j]) >= spacing for j in centers):
            centers.append(i)
        if len(centers) == fields:
            break
    if len(centers) < fields:
        raise ValueError('fewer separated fields than requested')
    targets = []
    groups = []
    for field, i in enumerate(centers):
        chosen = np.argsort(np.square(xy - xy[i]).sum(1), kind='stable')[:size]
        targets.extend(chosen.tolist())
        groups.extend([field] * size)
    if len(set(targets)) != fields * size:
        raise ValueError('overlapping fields')
    sort = np.argsort(targets)
    return np.asarray(targets)[sort], np.asarray(groups)[sort]


def distributed_recipients(ids, size=384):
    """The 384 cells with the smallest hash of the identifier."""
    order = sorted(
        range(len(ids)),
        key=lambda i: (hashlib.sha256((DISTRIBUTED_SALT + ids[i]).encode()).hexdigest(), ids[i]),
    )
    return np.array(order[:size], int)


def build_geometry(section, design):
    """Geometry arrays of one section for design 'fields' or 'distributed'."""
    xy = section[['Centroid_X', 'Centroid_Y']].to_numpy(float)
    ids = section.Cell_ID.to_numpy(str)
    labels, classes = annotations(section)
    if design == 'fields':
        target, groups = separated_fields(xy, ids)
    elif design == 'distributed':
        target = distributed_recipients(ids)
        groups = np.zeros(len(target), dtype=int)
    else:
        raise ValueError(design)
    return dict(
        coordinates=xy,
        ids=ids,
        target=target,
        groups=groups,
        labels=labels[target],
        class_labels=classes[target],
    )
