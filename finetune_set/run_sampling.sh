#!/usr/bin/env bash
# Run every generator on its assigned configurations. At most three GPU processes
# run at once; PET-OMat-XS runs on the CPU. Each finished configuration is saved,
# so the script can be rerun after an interruption.
cd "$(dirname "$0")/.."
E=~/anaconda3/envs
run() { "$E/$1/bin/python" finetune_set/sample.py --generator "$2" --threads "${3:-3}" >> "finetune_set/work/sampled/$2.log" 2>&1; }
mkdir -p finetune_set/work/sampled
run mace mace-mpa-0-medium &
(run grace grace-1l-oam && run mattersim mattersim-5m) &
run deepmd dpa-3.1-3m-omat24 &
run upet pet-omat-xs 4 &
wait
echo "sampling finished $(date)"
