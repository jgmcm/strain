# Fine-tune set: transition states

Transition-state frames for point-defect processes in graphene, from
climbing-image NEB with MACE-OMAT-0 small, scored by the 21-member committee of
the first set and reduced to 500–1,000 frames. The review pages are `index.qmd`
and `steps/*.qmd`; this README lists the commands.

## Steps

From the paper directory, with `E=~/anaconda3/envs`:

```sh
# 1. pairs of end states (2,000, seed 20261025)
$E/mace/bin/python finetune_set/ts/build_pairs.py
# 2. end-state relaxation and NEB rounds, batched on the GPU
$E/mace/bin/python finetune_set/ts/neb.py
# 3. candidate frames from the transition-state region of each usable band
(cd finetune_set/ts && $E/mace/bin/python assemble.py)
# 4. committee evaluation (MACE and DPA on the GPU; MatterSim, GRACE and PET on the CPU)
finetune_set/ts/run_committee.sh
# 5. selection, then figures and tables for the pages
$E/mace/bin/python finetune_set/ts/select_ts.py
$E/mace/bin/python finetune_set/ts/report_ts.py
```

`neb.py --endpoints-only` stops after the end-state relaxation. A restart reuses
`work/endpoints.extxyz` and reruns the NEB rounds. Settings are in
`config.json`. The scripts import helpers from the first set (`build_configs.py`,
`calculators.py`, `evaluate.py`, `select_frames.py`, `report.py`).

## GPU use

`neb.py` runs one GPU process, limited to half of the GPU memory, and pauses
after each batch for half the batch's run time (`--duty-pause 0.5`). The CPU
builds the next batch's graph during the pause. The committee evaluation runs
one GPU process at a time, with MatterSim, GRACE and PET on the CPU.

## Output

`finetune_ts_selected.extxyz` holds the selected frames and `ts_endpoints.extxyz`
the end states of their bands. Neither file contains reference energies or
forces; label the frames (for example with PBE single points) before
fine-tuning. See `steps/04_selection.qmd` for the per-frame fields.
