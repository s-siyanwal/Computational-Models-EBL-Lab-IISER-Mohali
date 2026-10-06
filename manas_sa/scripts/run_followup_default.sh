#!/usr/bin/env bash
# Follow-up heavy job: waits until the main default chain has finished
# (one heavy job at a time on the shared machine), then runs the LH-like power
# analysis and rebuilds the comparison table.
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python
until grep -q "chain exit" results/pipeline_default.log 2>/dev/null; do sleep 60; done
{
  date
  $PY scripts/run_power_lhlike.py --profile default
  $PY scripts/make_comparison.py > results/stage_table_followup.log 2>&1
  echo "followup exit $?"
  date
} > results/followup_default.log 2>&1
