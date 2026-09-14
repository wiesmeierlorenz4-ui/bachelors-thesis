#!/usr/bin/env bash
# FINAL 1500-run experiment — sequential orchestration.
#
# DO NOT run this without having explicitly approved the full paid run.
# This script sends real, paid OpenAI requests (Plain Direct, Direct-CoT,
# Code-Based Debug5 — 500 runs each, up to 3000 model generations for
# Code-Based alone, see README "API-Calls != experimentelle Läufe").
#
# Usage (from repo root, with caffeinate so macOS does not sleep mid-run):
#   caffeinate -i ./run_final_experiment.sh
#
# Requires: GRB_LICENSE_FILE set, OPENAI_API_KEY in .env, an OpenAI
# dashboard spend limit already configured (see README).
set -euo pipefail
cd "$(dirname "$0")"

export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/lizenzen/gurobi.lic}"

echo "############################################################"
echo "# 1/14  Preflight"
echo "############################################################"
python3 -m src.evaluation.final_preflight

echo
echo "############################################################"
echo "# 2/14  Plain Direct (SMTWTP + SICLSP, Batch API)"
echo "############################################################"
python3 -m src.chatgpt.run_direct          # both problems

echo
echo "############################################################"
echo "# 3/14  Validate Plain Direct completion (Direct-CoT/Code-Based still pending — partial FAIL expected here)"
echo "############################################################"
python3 -m src.evaluation.final_experiment_integrity_check || true

echo
echo "############################################################"
echo "# 4/14  Direct-CoT (SMTWTP + SICLSP, Batch API)"
echo "############################################################"
python3 -m src.chatgpt.run_direct_cot      # both problems (secondary experiment)

echo
echo "############################################################"
echo "# 5/14  Validate Direct-CoT completion (Code-Based still pending — partial FAIL expected here)"
echo "############################################################"
python3 -m src.evaluation.final_experiment_integrity_check || true

echo
echo "############################################################"
echo "# 6/14  Code-Based Debug5 (SMTWTP + SICLSP, synchronous API + sandbox)"
echo "############################################################"
python3 -m src.chatgpt.run_code_based      # both problems

echo
echo "############################################################"
echo "# 7/14  Validate Code-Based completion (all three treatments should now be complete)"
echo "############################################################"
python3 -m src.evaluation.final_experiment_integrity_check

echo
echo "############################################################"
echo "# 8-9/14  Evaluation + summary generation"
echo "############################################################"
python3 -m src.evaluation.build_summary_final

echo
echo "############################################################"
echo "# 10-13/14  Consolidated analysis layer (tables / per-run + per-instance datasets)"
echo "############################################################"
python3 analysis/final_results_summary/generate_final_results_summary.py

echo
echo "############################################################"
echo "# 14/14  Final integrity check"
echo "############################################################"
python3 -m src.evaluation.final_experiment_integrity_check

echo
echo "ALL STAGES COMPLETE."
