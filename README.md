# Strain engineering paper

Minimal Quarto research notebook, following the layout of `../../AgI/agi-hexatic`.
Start with `project.qmd`; shared methods belong in
`calculations/calculations.qmd`, and the first figure's notes belong in
`fig1_strain/notes.qmd`.

With Quarto on your PATH, run these commands from this directory:

```sh
quarto render
quarto preview --host 127.0.0.1 --no-browser
```

On this machine, the existing AgI installation can also be used:

```sh
../../AgI/agi-hexatic/.tools/bin/quarto render
../../AgI/agi-hexatic/.tools/bin/quarto preview --host 127.0.0.1 --no-browser
```

The rendered notebook starts at `_site/index.html`. Code execution is disabled;
rendering the notes requires no calculation environment.

The first local calculation setup is in `calculations/01_graphene/`: an ASE
script generates an `n x n` graphene sheet, and a 4 x 4 example structure is
included. See the calculation README for commands and environment setup.
The first local MACE-MPA-0 relaxations are complete in
`calculations/02_vacancy_relaxation/`: 4 x 4 and 5 x 5 graphene with two vacancies
each, fixed periodic cells, and fully relaxed atomic coordinates. Figure 1 shows
the in-plane displacement field. The calculation README records settings,
reproduction commands, checkpoint provenance and the limits of this first pass.

The larger 20 x 20 and 21 x 21 set is complete in
`calculations/03_vacancy_relaxation_20_21/`, using the same model and relaxation
settings. Its plotting script opens the vector map and the Gouraud-shaded
magnitude map with contour lines. See the calculation README for results and
the sublattice difference between the two vacancy patterns.

The formation-energy scan for two vacancies in 2 x 2 through 10 x 10 cells is
complete in `calculations/04_vacancy_formation_energy_2_10/`. It uses the perfect
two-carbon graphene unit cell as the carbon chemical-potential reference and
reports both the total energy of two vacancies and the average per vacancy.
Its README and combined CSV contain the results; the plotting script opens
the formation-energy figure with `plt.show()`.
