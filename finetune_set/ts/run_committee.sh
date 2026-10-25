#!/usr/bin/env bash
# Evaluate the 21 committee members on the transition-state candidates.
# Two chains run side by side: MACE then DPA on the GPU, and MatterSim, GRACE and PET one after another on the CPU,
# with small CPU batches to keep system memory low. Start it as a memory-capped user service, for example
#   systemd-run --user --unit=ts-committee --collect -p MemoryMax=14G finetune_set/ts/run_committee.sh
# Finished members are skipped, so the script can be rerun after an interruption.
cd "$(dirname "$0")/../.."
E=~/anaconda3/envs
POOL=finetune_set/ts/work/candidates.extxyz
OUT=finetune_set/ts/work/committee
LOG=finetune_set/ts/work/committee.log
ev() { "$E/$1/bin/python" finetune_set/evaluate.py --env "$1" --pool "$POOL" --output "$OUT" "${@:2}" >> "$LOG" 2>&1 \
         || echo "evaluation failed for $1" >> "$LOG"; }
GPU="--duty-pause ${DUTY_PAUSE:-0.5} --max-atoms ${MAX_ATOMS:-2000}"
CPU="--device cpu --duty-pause 0 --max-atoms 300 --threads 6"
(ev mace $GPU; ev deepmd $GPU) &
(ev mattersim $CPU; ev grace $CPU; ev upet $CPU) &
wait
echo "committee finished $(date)" >> "$LOG"
