# Fine-tune set: committee-selected graphene structures

Graphene structures with strain, curvature and point defects, selected by
committee disagreement for fine-tuning an interatomic potential. The review
pages are `index.qmd` and `steps/*.qmd`; this README lists the commands.

## Environments

| Conda environment | Members | Notes |
|---|---|---|
| `mace` | 16 MACE configurations | existing |
| `mattersim` | MatterSim 1M, 5M | existing |
| `upet` | PET-OMat-XS v1.0.0 | new: `upet` 0.3.1 |
| `deepmd` | DPA-3.1-3M, `Omat24` head | new: `deepmd-kit[torch]` 3.1.3 |
| `grace` | GRACE-1L-OAM | new: `tensorpotential` 0.6.2.1 |

The new environments were created from conda-forge (Python 3.12) and the
packages installed with `uv pip`. The MACE and MatterSim checkpoints come from
`../comp_models/.models/`; the new ones are in `.models/` (ignored by git). The
DPA head was frozen with
`dp --pt freeze -c DPA-3.1-3M.pt -o DPA-3.1-3M-Omat24.pth --model-branch Omat24`.
Checksums are in `committee.json`.

## Steps

From the paper directory:

```sh
E=~/anaconda3/envs
# 1. starting configurations (1,300, seed 20261025)
$E/mace/bin/python finetune_set/build_configs.py
# 2. relaxation and MD with the five generators (resumable)
finetune_set/run_sampling.sh
# 3. candidate pool and filters
$E/mace/bin/python finetune_set/assemble.py
# 4. committee evaluation, one member at a time (resumable)
finetune_set/run_committee.sh
# 5. selection, then figures and tables for the pages
$E/mace/bin/python finetune_set/select_frames.py
$E/mace/bin/python finetune_set/report.py
```

`sample.py --generator <key>` runs one generator; `evaluate.py --env <env>`
evaluates all members in one environment, with `--members` to select some.
Both skip work that is already saved.

## GPU use

Sampling runs at most three generator processes on the GPU at once and runs PET
on the CPU. Committee evaluation runs one GPU process at a time, limits it to
half of the GPU memory, and pauses after each batch for `--duty-pause` times the
batch's run time. The run on 2026-10-25 used `DUTY_PAUSE=1.0 MAX_ATOMS=3000
finetune_set/run_committee.sh`, with MatterSim, PET and GRACE evaluated on the
CPU in parallel (`evaluate.py --env <env> --device cpu --duty-pause 0`).
`evaluate.py` writes a lock file while it evaluates a member, so a second
process skips that member.

## Output

`finetune_selected.extxyz` holds the selected frames. It contains structures
and committee statistics but no reference energies or forces; label the frames
(for example with PBE single points) before fine-tuning. See
`steps/05_selection.qmd` for the per-frame fields.
