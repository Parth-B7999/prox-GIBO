#!/bin/bash
# Extra demo runs (GP-family methods only), executed one config after another with 2 workers,
# alongside the main grid. Each config's runs are skipped if already present.
cd "$(dirname "$0")/.."
for c in config_d200 config_noise0 config_stress_cond100; do
  echo "=== $c $(date +%H:%M:%S)"
  conda run --no-capture-output -n nestbo python experiments/run_demo.py experiments/$c.yaml
done
echo "=== extras finished $(date +%H:%M:%S)"
