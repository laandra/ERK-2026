#!/bin/bash
# The hyperparameter search under the wide contract. Resume = run this again.
cd /Users/summerscholl/Documents/ERK-2026/Main
export ERK_HPO_ROOT=$PWD/results_local/rl_hpo_wide
export ERK_HPO_MODELS=$PWD/rl_models/hpo_wide
export ERK_HPO_VARIANT=wide_h24
export ERK_HPO_OBJECTIVE=shift
exec python3 run_rl_hpo.py "$@"
