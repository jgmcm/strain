# Local calculation 02: vacancy relaxation

Both periodic graphene vacancy cells have been relaxed with the locally available
**MACE-MPA-0 medium** foundation model, using the existing
`/opt/anaconda3/envs/mace` environment. This is an exploratory model calculation,
not a DFT-validated vacancy calculation.

## Method

- 4 x 4: remove original ASE atom indices 0 and 19, leaving 30 carbon atoms.
- 5 x 5: remove original ASE atom indices 0 and 17, leaving 48 carbon atoms.
- Keep cell vectors fixed at a lattice constant of 2.46 Angstrom, with 15 Angstrom
  vacuum per side and periodicity in-plane only. All atomic coordinates relax in 3D.
- Add a zero-mean Gaussian displacement of 0.01 Angstrom standard deviation before
  relaxation, with seed `20261010 + n`, to allow symmetry breaking.
- ASE BFGS, maximum step 0.1 Angstrom, convergence at maximum all-atom force below
  0.005 eV/Angstrom; CPU, four Torch threads, float64, no added dispersion correction.

`config.json` records the settings. The pinned checkpoint is copied under the
ignored `.models/` directory; `model_provenance.json` records its local source,
official release URL and SHA-256. `environment.json` records package versions.
MACE-MATPes-PBE was also found locally; it was not used in this first pass.

## Results

| Cell | BFGS steps | Final max force (eV/Angstrom) | Max in-plane displacement (Angstrom) | RMS in-plane displacement (Angstrom) |
|---|---:|---:|---:|---:|
| 4 x 4 | 22 | 0.003587 | 0.055897 | 0.040583 |
| 5 x 5 | 18 | 0.004394 | 0.082216 | 0.039720 |

Energy changes from the ideal defective structures are -0.527837 and -0.473429 eV,
respectively. These are relaxation energy changes, not vacancy formation energies.
The relaxed z ranges are 0.001659 and 0.007727 Angstrom.

## Displacement definition

Match each surviving atom to the same atom in the ideal, unperturbed defective
structure. Compute the minimum-image displacement with ASE's `find_mic`, then
subtract its mean vector to remove rigid translation. The initial random
perturbation is excluded from the reference.

The primary plotted quantity is `u_xy_magnitude = sqrt(ux**2 + uy**2)`, in Angstrom.
The CSV also includes `uz`, the full 3D magnitude, atom identity, reference and
relaxed coordinates, and reference distance to the nearest periodic vacancy.
No strain tensor or strain trace has been calculated. The second displayed figure
uses Matplotlib `tripcolor(..., shading="gouraud")` on a triangulation of the
original atom magnitudes, with `tricontour` lines overlaid. Periodic images are
included before triangulation and the result is clipped to one cell. Triangles
containing missing sites are masked. Gouraud shading interpolates vertex colours;
the atomic displacement values are used directly.
The triangulation resolves co-circular lattice sites consistently in periodic
copies; its plotted vertex coordinates and atom values are the original data.

## Files and reproduction

Run from the paper directory:

```sh
/opt/anaconda3/envs/mace/bin/python calculations/02_vacancy_relaxation/relax.py
/opt/anaconda3/envs/mace/bin/python calculations/02_vacancy_relaxation/plot_displacement.py
```

The plotting script saves the figures, then opens two in Matplotlib windows
with `plt.show()`. Close the windows to return to the terminal.
Use `--sizes 4 5` to select cell directories, or `--no-show` to save without windows.

The relaxation script keeps already-converged results with matching settings.
Use `--resume` for an unfinished run; `--sizes 4` or `--sizes 5` selects one size.
For different settings, start a new calculation directory to preserve this run.

Each `4x4/` or `5x5/` directory contains:

- `reference.extxyz`: ideal surviving sites, before the random perturbation.
- `initial.extxyz`: perturbed starting structure.
- `relaxed.extxyz`: final structure, forces, energy and displacement arrays.
- `vacancy_sites.extxyz`: two removed positions marked with dummy X atoms, for inspection.
- `relax.log`, `relax.traj`, `optimizer.json`: optimisation record and restart.
- `settings.json`, `metrics.json`, `displacement.csv`: method and numerical results.

The main figure is `displacement_maps.png` (also SVG/PDF): colour shows in-plane
magnitude and arrows show `(ux, uy)` magnified by five. The second figure,
`displacement_contours.png`, uses Gouraud shading with contour lines,
using the same colour scale. Both figures are available as PNG/SVG/PDF.
