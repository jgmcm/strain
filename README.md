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

The foundation-model comparison is in `comp_models/`. Seventeen checkpoints
(MACE-MP-0, MP-0b2/0b3, MPA-0, OMAT-0, MATPES-0, MH-0/1, OFF23 and MatterSim)
are stored in its ignored `.models/` directory, with provenance in
`model_provenance.json`. The calculation repeats the 4 x 4 and 5 x 5 relaxation
of calculation 02 with 19 model configurations and computes per-atom and
per-bond strain. Its page `comp_models/notes.qmd` has a model browser for the
Figure 1 Gouraud displacement maps, all models side by side, summary tables
and strain maps; its README has the method and reproduction commands.

The fine-tune set is in `finetune_set/`. Five foundation models generated
23,731 candidate frames of strained, curved and defective graphene, nanotubes
and C60 (at most 100 atoms each); a 21-member committee scored them, and 1,507
frames with the largest disagreement, kept structurally diverse, are in
`finetune_set/finetune_selected.extxyz`. The page `finetune_set/index.qmd` and
its step sub-pages document each stage; `finetune_set/README.md` has the
commands.
