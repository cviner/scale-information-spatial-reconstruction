# Preserving scale information in molecular spatial reconstruction

## Installation

```
pip install -r requirements.txt
```

## Checking the saved fits

```
python verify_examples.py
```

Checks the saved fits and recomputed metrics for the four supplied cases against `results/`. It does not refit the models, and any mismatch stops it with an assertion error.

The printed summary includes two diagnostics from Section 3.5 of the paper. `allocation_gain_on_better_placed_pairs_pct` is the share of the gross positive allocation gain of joint P over the calibrated pooled shape that comes from pairs whose aligned distance error also improved. `within_field_pairs_placed_closer_pct` is the percentage of within-field pairs with that improvement.

## Regenerating simulated assays

```
python verify_examples.py --simulate
```

Also regenerates the four assays of the first case from their seeds and compares them with the saved inputs.

## Running a case

```
python run_case.py --geometry data/fields/animal1_geometry.npz --seed 22701 --design mix25 --out work/seed22701_animal1_mix25
```

Simulates the four assays of one acquisition, fits them, and scores the maps. Refits can differ slightly from the saved fits. `--out` must be a new folder. It receives the layout of the cases in `examples/`, with the scores in `scores.json`.

- `--geometry`: a file in `data/fields/` for separated fields, or in `data/distributed/` for distributed recipients.
- `--design`: `r15` to `r480`, the twelve single transport radii in micrometres, or `mix25`, in which each packet takes 15 µm with probability 1/4 and 480 µm otherwise.
- `--seed`: the paper used 22701 and 22702 for animal 1, and 22703 and 22704 for animal 2.
- `--shapes`: also calibrates the 23 comparison shapes.

## Result tables

The saved maps are included only for the four example cases, and the incidence matrices are not included. `code/simulate.py` regenerates the incidence from the seeds.

- `seed` is the technical replicate.
- `pooled` is the pooled competitor after its continuations, and `pooled_before_continuation` is the best of its six starts.
- `full`, `total`, and `allocation` are deviances per observed evaluation count.
- `message` is the termination message of SciPy's L-BFGS-B.
- In `distance_bands.csv`, the row with empty bounds covers all pairs in scope.
- `results/fields/composition_without_position.csv` is the composition error when every recipient is given the leave-one-out composition of all recipients.
- `results/budgets.csv` gives means over the two assays of each split.
- `results/channel_response.csv` and `channel_contribution.csv` give animal means of the two replicates, in bands of reference distance.

## Data

The tissue geometry comes from the MERFISH cell table of Moffitt, Bambah-Mukku et al. (*Science* 2018, [doi:10.1126/science.aau5324](https://doi.org/10.1126/science.aau5324)), deposited in Dryad ([doi:10.5061/dryad.8t8s248](https://doi.org/10.5061/dryad.8t8s248)) under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

- `data/sections/`: the rows of the table for the two selected sections, with repeated records at one centroid reduced to one, and the number of cells in every section of the table (`all_sections.csv`).
- `data/fields/` and `data/distributed/`: the recipients of the two designs, derived from those sections by `code/geometry.py`.

To repeat the selection from the full table, use `code/geometry.py`.

## Licence

The code is under the GNU General Public License, version 3 (see `LICENSE`). The GPL does not cover `data/`, which is under CC0 1.0.
