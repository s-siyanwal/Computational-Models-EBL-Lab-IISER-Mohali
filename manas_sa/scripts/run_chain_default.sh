#!/usr/bin/env bash
# Detached long run of the default profile (several hours on a 4-core laptop).
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
{
  date
  $PY scripts/run_all.py --profile default --only design,emulator &&
  $PY scripts/run_lambda_check.py --maps 0.1 --lams 1 --reps 6 > results/lambda_check_lam1.log 2>&1 &&
  $PY scripts/run_all.py --profile default --only hri,distinguish,table
  echo "chain exit $?"
  date
} > results/pipeline_default.log 2>&1
