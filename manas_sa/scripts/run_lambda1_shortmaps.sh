#!/usr/bin/env bash
# Audit fix: validate the post-hoc collapse filter at full scale (lambda = 1) for the short maps.
cd "$(dirname "$0")/.."
../.venv/Scripts/python scripts/run_lambda_check.py --maps 0.001,0.01 --lams 1 --reps 6 > results/lambda1_shortmaps.log 2>&1
echo "lambda1 exit $?" >> results/lambda1_shortmaps.log
