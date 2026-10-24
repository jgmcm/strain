# Local calculation 01: graphene

First step: generate an unstrained graphene sheet. A 4 x 4 example structure is
included. No calculator or relaxation is attached yet.

From the paper directory (`strain-engineering/`), with ASE installed:

```sh
python calculations/01_graphene/make_graphene.py 4
```

The existing AgI environment is also available on this machine:

```sh
../../AgI/agi-hexatic/.venv/bin/python calculations/01_graphene/make_graphene.py 4
```

The script writes `graphene_4x4.extxyz` beside itself, containing 32 carbon atoms.
It uses [ASE's graphene builder](https://docs.ase-lib.org/_modules/ase/build/surface.html#graphene)
to repeat the two-atom primitive cell `n` times along each in-plane lattice vector.
The resulting cell is hexagonal (in-plane angle 120 degrees), with periodic
boundaries in x/y and a nonperiodic z direction.

Defaults are a lattice constant of 2.46 Angstrom and 15 Angstrom of vacuum on
each side (30 Angstrom cell height). The lattice constant is an initial geometry
choice, not a relaxed result. Extended XYZ preserves the cell and periodicity.

Override the defaults or destination as needed:

```sh
python calculations/01_graphene/make_graphene.py 6 --a 2.46 --vacuum 20 \
  --output calculations/01_graphene/graphene_6x6.extxyz
```

For a separate environment, from the paper directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Validated with ASE 3.29.0.

## Explicit vacancy-superlattice check

`check_repeated_vacancies.py` builds actual 4 x 4 and 5 x 5 atom cells, marks
two carbon sites, and repeats each cell 7 x 7. It checks ordinary Cartesian
distances to all 98 explicit vacancy sites, with periodic wrapping disabled.
Both vacancy types in all 25 interior cells (50 sites per size) have exactly
three equally distant nearest vacancies and three pairwise angles of 120 degrees.

The original, zero-based vacancy indices are `(0, 19)` for 4 x 4 and `(0, 17)`
for 5 x 5. Nearest distances are 5.681126649 and 7.101408311 Angstrom,
respectively, at a lattice constant of 2.46 Angstrom. The fourth-nearest
vacancies are farther away: 9.84 and 12.30 Angstrom.

From the paper directory:

```sh
../../AgI/agi-hexatic/.venv/bin/python calculations/01_graphene/check_repeated_vacancies.py
```

The script saves full coordinates and numerical checks in
`explicit_repeated_vacancy_check.json`. The two
`graphene_*_two_vacancies_repeated7x7.extxyz` files contain the finite repeated
carbon sheets; the separate `vacancy_sites_*` files mark removed sites with
dummy `X` atoms for inspection. These files are a geometry check, with no
calculator or relaxation.
