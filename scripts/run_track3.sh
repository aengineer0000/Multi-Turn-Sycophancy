#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
MODEL_NAME="${MODEL_NAME:-google/gemma-2-2b-it}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

mkdir -p data/processed reports/figures reports/tables

if [[ "$MODE" == "smoke" ]]; then
  echo "[1/3] Running 5-example behavioral smoke test..."
  python src/eval/behavioral_baseline.py \
    --model "$MODEL_NAME" \
    --limit 5

  echo "[2/3] Scoring smoke-test responses with heuristic scorer..."
  python src/eval/score_behavior.py

  echo "[3/3] Running full dataset confounder audit..."
  python src/eval/audit_confounders.py \
    --model "$MODEL_NAME"

  echo
  echo "Smoke test complete. Inspect data/processed/behavioral_results.json before a full run."
  exit 0
fi

if [[ "$MODE" == "full" ]]; then
  echo "[1/3] Running full behavioral baseline..."
  python src/eval/behavioral_baseline.py \
    --model "$MODEL_NAME" \
    --resume

  echo "[2/3] Scoring full responses..."
  if [[ -n "${GOOGLE_API_KEY:-}" && "${GOOGLE_API_KEY}" != "replace_me" ]]; then
    python src/eval/score_behavior.py --use-llm-judge
  else
    echo "GOOGLE_API_KEY not configured; using heuristic scorer only."
    python src/eval/score_behavior.py
  fi

  echo "[3/3] Running confounder audit..."
  python src/eval/audit_confounders.py \
    --model "$MODEL_NAME"

  echo
  echo "Full Track 3 pipeline complete."
  exit 0
fi

echo "Usage: bash scripts/run_track3.sh [smoke|full]"
exit 2
