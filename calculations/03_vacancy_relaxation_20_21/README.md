# Local calculation 03: 20 x 20 and 21 x 21 vacancy cells

Larger cells using the same MACE-MPA-0 medium checkpoint, fixed lattice constant
2.46 Angstrom, 15 Angstrom vacuum per side, in-plane periodicity and 3D atomic
relaxation as calculation 02. ASE BFGS uses a maximum step of 0.1 Angstrom and
maximum-force convergence below 0.005 eV/Angstrom. The reproducible starting
perturbation is Gaussian, 0.01 Angstrom standard deviation, seed `20261010 + n`.
Calculations use CPU, four Torch threads and float64.

## Vacancy geometry

Both vacancy sets are at full-cell fractional positions `(0, 0)` and `(1/3, 2/3)`.

| Cell | Original ASE vacancy indices | Parent carbon sublattices | Remaining carbon atoms | Nearest vacancy distance (Angstrom) |
|---|---|---|---:|---:|
| 20 x 20 | 0, 267 | A, B | 798 | 28.405633 |
| 21 x 21 | 0, 322 | A, A | 880 | 29.825915 |

Indices are zero-based, before deleting either atom. The 21 x 21 pattern requires
same-sublattice vacancies for the exact two-site honeycomb superlattice; the two
cases therefore differ in parent sublattice arrangement as well as cell size.
An explicit 7 x 7 repetition verified three equal nearest distances and three
120-degree angles at all 50 interior vacancy sites for each cell.
See `vacancy_geometry_check.json` for the numerical record.

## Reproduction

From the paper directory:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  /opt/anaconda3/envs/mace/bin/python calculations/03_vacancy_relaxation_20_21/relax.py
/opt/anaconda3/envs/mace/bin/python calculations/03_vacancy_relaxation_20_21/plot_displacement.py
```

Use `--sizes 20` or `--sizes 21` to select one case. The plotting script opens
the displacement vector map and the Gouraud-shaded magnitude map with contour
lines. Use `--no-show` to export without windows. There is no Gaussian smoothing
of the plot data. The denser vector plots use smaller atom markers and narrower
arrows, retaining fivefold arrow magnification.

The pinned checkpoint is reused from calculation 02's `.models/` directory.
`model_provenance.json` records its SHA-256; `config.json` and `environment.json`
record settings and versions. BLAS and OpenMP thread counts are limited to four
for the larger dense BFGS operations.

Each cell directory contains reference, perturbed and relaxed structures,
vacancy markers, an optimisation log/trajectory/restart, and final metrics and
atom-resolved displacement CSV. Displacement is measured against ideal surviving
sites with minimum-image wrapping and rigid translation removed. The primary
quantity is `sqrt(ux**2 + uy**2)`; no strain tensor is calculated.

## Results

Both cells converged and passed fresh MACE evaluations of their saved structures.

| Cell | BFGS steps | Final max force (eV/Angstrom) | Max in-plane displacement (Angstrom) | RMS displacement (Angstrom) |
|---|---:|---:|---:|---:|
| 20 x 20 | 89 | 0.004324 | 0.334622 | 0.046582 |
| 21 x 21 | 29 | 0.004811 | 0.106652 | 0.015206 |

The relaxation energy changes from ideal defective structures are -0.919435 eV
(20 x 20) and -0.602114 eV (21 x 21). The relaxed z ranges are 0.009392 and
0.012343 Angstrom. These are fixed-cell, force-converged local structures.

The 20 x 20 result forms a close neighbour pair of approximately 1.813 Angstrom
at each vacancy. In the 21 x 21 result, the three original nearest-carbon pair
distances remain approximately 2.28-2.30 Angstrom. The two runs therefore reach
different local vacancy geometries at the selected force tolerance, alongside
their different parent sublattice arrangements.

See `validation.json` for force and periodic-plot checks, and each cell
directory's `local_vacancy_geometry.json` for neighbour-pair distances. The
figures are `displacement_maps` and `displacement_contours`, available as PNG,
SVG and PDF. Both cases share the same colour scale.
