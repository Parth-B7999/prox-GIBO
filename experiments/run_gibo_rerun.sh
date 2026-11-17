#!/bin/bash
# Rerun GIBO presets after the single-point GP fit fix (2026-11-15). Old results: results/_invalid_gp_init_bug/
cd "$(dirname "$0")/.."
echo "=== demo gibo $(date +%T)"
conda run --no-capture-output -n nestbo python experiments/run_demo.py experiments/config_d200.yaml --only gibo --dims 50 100 200 --workers 3
echo "=== gibo_paper $(date +%T)"
conda run --no-capture-output -n nestbo python experiments/run_demo.py experiments/config_gibo_paper.yaml --workers 3
echo "=== noise0 gibo $(date +%T)"
conda run --no-capture-output -n nestbo python experiments/run_demo.py experiments/config_noise0.yaml --only gibo --workers 3
echo "=== gibo rerun finished $(date +%T)"
