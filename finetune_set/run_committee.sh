#!/usr/bin/env bash
# Evaluate all 21 committee members on the candidate pool, one GPU process at a time.
# Finished members are skipped, so the script can be rerun after an interruption.
cd "$(dirname "$0")/.."
E=~/anaconda3/envs
for env in mace mattersim upet deepmd grace; do
  "$E/$env/bin/python" finetune_set/evaluate.py --env "$env" --duty-pause "${DUTY_PAUSE:-0.5}" --max-atoms "${MAX_ATOMS:-2000}" \
    >> finetune_set/work/committee.log 2>&1 || echo "evaluation failed for $env" >> finetune_set/work/committee.log
done
echo "committee finished $(date)" >> finetune_set/work/committee.log
