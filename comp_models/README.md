# Model comparison: two-vacancy 4 x 4 and 5 x 5 graphene

The calculation 02 relaxation repeated with 19 foundation-model configurations:
15 MACE checkpoints (two of them with two heads each) and two MatterSim
checkpoints. The page `notes.qmd` summarises the results; this README records
the method, files and reproduction commands.

## Model weights

`models.json` is the registry. Its `checkpoints` list gives each weight file,
official URL, upstream size, licence and, where upstream publishes one, the
expected digest. Its `models` list defines the run configurations: checkpoint,
conda environment, MACE head, training data and level of theory.

`fetch_models.py` copies each checkpoint from a local cache when one with the
upstream size exists, otherwise downloads it, into the ignored `.models/`
directory (1.1 GB). It checks every file's size against the upstream release,
and the SHA-256 or git blob digest where upstream publishes one
(MACE-MPA-0, MACE-OMAT-0 small, MACE-MH-0/1, MatterSim 1M/5M).
`model_provenance.json` records the local path, URL, licence, size and SHA-256
of each file; `relax.py` refuses to run if a checkpoint no longer matches it.

The MACE-MP-0 large file is `2024-01-07-mace-128-L2_epoch-199.model`, the file
named in the MACE-MP-0 release notes. It has 5,725,072 parameters, matching
those notes. MACE 0.3.16's `mace_mp(model="large")` downloads a different file,
`MACE_MPtrj_2022.9.model`, which is not used here.

Licences: MACE-MP-0, MACE-MP-0b2/0b3, MACE-MPA-0 and MatterSim are MIT;
MACE-OMAT-0, MACE-MATPES-0, MACE-MH-0/1 and MACE-OFF23 are under the Academic
Software License (ASL).

## Method

The structures, perturbation and optimiser match calculation 02:

- 4 x 4: remove original ASE atom indices 0 and 19 (30 carbon atoms remain).
- 5 x 5: remove original ASE atom indices 0 and 17 (48 carbon atoms remain).
- Fixed cell at a = 2.46 Angstrom, 15 Angstrom vacuum per side, in-plane
  periodicity; all atomic coordinates relax in 3D; no dispersion correction.
- Zero-mean Gaussian starting perturbation, standard deviation 0.01 Angstrom,
  seed `20261010 + n`. Every model starts from identical coordinates.
- ASE BFGS, maximum step 0.1 Angstrom, maximum force below 0.005 eV/Angstrom,
  at most 1000 steps per invocation; CPU, four Torch threads.

MACE models run in float64. MatterSim 1.2.1 builds float32 graph tensors and
fails with float64 parameters, so both MatterSim checkpoints run in float32.
MACE-MH-0 and MACE-MH-1 run twice, once with the `omat_pbe` head and once with
the `matpes_r2scan` head. The script checks that the requested head was loaded,
because MACE falls back to another head with only a warning.

After each relaxation the saved structure is re-read and evaluated again; the
run passes if the fresh maximum force is below tolerance and the energy agrees
with the saved value (1e-6 eV in float64, 1e-4 eV in float32).

Each model also gets an equilibrium lattice constant `a_eq` from a bounded
one-dimensional minimisation of the perfect two-atom cell energy per atom
(`runs/<model>/lattice.json`). The fixed cell then imposes a uniform biaxial
strain `2.46/a_eq - 1` relative to that model's own equilibrium.

## Strain definitions

`analyse_strain.py` uses the ideal surviving sites in the fixed cell as the
reference and minimum-image displacements, as in calculation 02.

- Atomic strain: for each atom, the least-squares in-plane deformation gradient
  `F` mapping its same-sublattice (second-neighbour, length a) reference vectors
  onto the relaxed vectors. Same-sublattice vectors exclude the internal shift
  between the two graphene sublattices. Atoms have six such neighbours, or five
  when one is a vacancy site. Green-Lagrange strain `E = (F^T F - I)/2`.
- Areal strain `det F - 1`, the relative change of local area.
- Maximum shear strain `(E_1 - E_2)/2` from the principal values of `E`.
- Bond strain `l/l0 - 1` for every surviving reference C-C bond, with 3D lengths.
- Vacancy-neighbour distances: the three pairwise distances between the
  three carbon atoms around each vacancy (2.46 Angstrom in the ideal lattice).
  One short pair indicates a reconstructed vacancy.

The estimator self-test recovers zero strain for zero displacement and an
imposed homogeneous deformation gradient to 1e-12 at every atom, including
atoms beside vacancies (`validation.json`).

## Reproduction

From the paper directory:

```sh
python3 comp_models/fetch_models.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  ~/anaconda3/envs/mace/bin/python comp_models/relax.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  ~/anaconda3/envs/mattersim/bin/python comp_models/relax.py
~/anaconda3/envs/mace/bin/python comp_models/analyse_strain.py
~/anaconda3/envs/mace/bin/python comp_models/plot_strain.py
~/anaconda3/envs/mace/bin/python comp_models/plot_displacement.py
```

`relax.py` runs every registered model for the environment it is started in;
`--models` and `--sizes` select a subset. Converged runs with matching
settings are kept. `--resume` continues an unfinished run; adding
`--reset-hessian` continues from the last frame with a fresh BFGS Hessian, and
the reset step is recorded in `metrics.json` as `bfgs_hessian_reset_steps`.

`plot_displacement.py` draws the calculation 02 Gouraud contour figure for
every model, once on a colour scale shared by all models (0 to the largest
displacement among runs with a symmetric vacancy, rounded up to 0.01 Angstrom)
and once on the model's own scale. It saves these in `figures/displacement/`,
draws all models on the shared scale in `displacement_contours_all_<n>x<n>`
(PNG/PDF), and writes the model browser `_displacement_browser.md`. It opens
the two all-model figures with `plt.show()`.

`plot_strain.py` opens the overview figure with `plt.show()`; use `--no-show`
with either script to save without a window. It also writes the map gallery and `_strain_maps.md`,
and `analyse_strain.py` writes `_summary_tables.md`; `notes.qmd` includes both.

## Run notes

MACE-MP-0b2 small in the 5 x 5 cell stalled: from about step 250 to step 1000,
the energy stayed constant to 1e-6 eV with a maximum force of 0.151 eV/Angstrom.
Central finite differences along the largest force reproduced it to 1e-5
eV/Angstrom, so the force is the energy gradient; the BFGS Hessian estimate had
degenerated. `--resume --reset-hessian` continued from the step-1000 frame
with a fresh Hessian and converged at step 1046.

That resume wrote `relaxed.extxyz` without the `original_site_index` array,
because ASE trajectory frames do not store custom arrays. `relax.py` now
restores the array on resume. The array was added to the one affected file
from `reference.extxyz`, after checking that its positions match the final
trajectory frame to 5e-9 Angstrom. Calculation 02's `relax.py` has the same
resume path.

## Results

All 38 runs converged and passed the fresh evaluation, and the MACE-MPA-0 runs
reproduce calculation 02 exactly. MACE-MP-0b2 small (both cells) and
MACE-MP-0b3 medium (5 x 5) reconstruct the vacancy, forming a 1.815-1.819
Angstrom pair between two vacancy neighbours, with areal strain up to +8.8 %.
The other materials models keep a symmetric vacancy. Their vacancy-neighbour
distances range from 2.19 to 2.55 Angstrom, and their areal strain from -4.3 %
(MACE-MP-0 small) to within +/-1.5 % (OMAT-0, MATPES-0, MH-0/1, MatterSim 5M).
Equilibrium lattice constants range from 2.4512 to 2.4778 Angstrom across the
materials models. See `notes.qmd` and `summary.csv`.

## Files

- `runs/<model>/lattice.json`: equilibrium lattice constant and fixed-cell strain.
- `runs/<model>/<n>x<n>/`: `settings.json`, `reference.extxyz`, `initial.extxyz`,
  `relaxed.extxyz`, `vacancy_sites.extxyz`, `relax.log`, `metrics.json`,
  `displacement.csv`, `strain.csv`, `bond_strain.csv`, `vacancy_geometry.json`.
  `relax.traj` and `optimizer.json` are kept locally but ignored by git.
- `summary.csv`, `summary.json`: one row per model and cell.
- `validation.json`: estimator self-test and the comparison with calculation 02.
- `strain_distributions.png` (also SVG/PDF): per-atom and per-bond strain for all models.
- `figures/`: strain maps by quantity and model family (PNG).
- `figures/displacement/`: Gouraud contour figures per model, `common_<model>.png` and `own_<model>.png`.
- `displacement_contours_all_4x4`, `displacement_contours_all_5x5` (PNG/PDF): all models on the shared scale.
- `environment_mace.json`, `environment_mattersim.json`: package versions.
