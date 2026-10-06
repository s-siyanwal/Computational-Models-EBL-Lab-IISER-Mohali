#!/usr/bin/env bash
cd "$(dirname "$0")/.."
../.venv/Scripts/python scripts/run_ne_check.py --Ns 625,1250 --reps 6 > results/ne_check.log 2>&1
echo "ne exit $?" >> results/ne_check.log
