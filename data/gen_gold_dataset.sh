#!/usr/bin/env bash
# =============================================================================
# gen_gold_dataset.sh — build the GOLD dataset (the profile-mix sweep).
#
# For each profile mix (single / multi_turn / agentic percentages) it:
#   1. derives a config from configs/gold_config/base.yaml  (write_mix_config.py)
#   2. generates that config's trace                        (mixed_workload_gen_ai_perf.py)
#
# Configs land in  configs/gold_config/s<S>_m<M>_a<A>.yaml
# Traces  land in  data/gold/traces/s<S>_m<M>_a<A>.jsonl
#
# USAGE   (run from anywhere)
#   ./data/gen_gold_dataset.sh
# ENV (optional)
#   PYTHON   interpreter with numpy + tokenizers   (default: python3)
#   SEED     empty = use the base config's seed; set an int to override
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOL="$REPO_ROOT/tools/workload_generator"
BASE="$REPO_ROOT/configs/gold_config/base.yaml"
CFGDIR="$REPO_ROOT/configs/gold_config"
OUTDIR="$REPO_ROOT/data/gold/traces"

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-}"
mkdir -p "$CFGDIR" "$OUTDIR"

# percent triples "single multi agentic" — the 21-point ternary simplex
COMBOS=(
  "100 0 0"  "70 30 0"  "70 0 30"  "50 50 0"  "50 0 50"  "30 70 0"  "30 0 70"
  "0 100 0"  "0 70 30"  "0 50 50"  "0 30 70"  "0 0 100"
  "70 20 10" "70 10 20" "40 30 30" "30 40 30" "30 30 40"
  "20 70 10" "10 70 20" "20 10 70" "10 20 70"
)

for combo in "${COMBOS[@]}"; do
  read -r s m a <<< "$combo"
  name="s${s}_m${m}_a${a}"
  cfg="$CFGDIR/${name}.yaml"
  out="$OUTDIR/${name}.jsonl"

  # percent -> fraction (kept exact-ish so the sum-to-1.0 check passes)
  fs=$(awk "BEGIN{printf \"%.10g\", $s/100}")
  fm=$(awk "BEGIN{printf \"%.10g\", $m/100}")
  fa=$(awk "BEGIN{printf \"%.10g\", $a/100}")

  "$PYTHON" "$TOOL/write_mix_config.py" \
      --base "$BASE" --out "$cfg" \
      --single "$fs" --multi "$fm" --agentic "$fa"

  echo "=== single=${s}% multi=${m}% agentic=${a}%  ->  $out ==="
  cmd=("$PYTHON" "$TOOL/mixed_workload_gen_ai_perf.py" --config "$cfg" --out "$out")
  [ -n "$SEED" ] && cmd+=(--seed "$SEED")
  "${cmd[@]}"
done

echo "done: $(ls "$OUTDIR"/*.jsonl 2>/dev/null | wc -l | tr -d ' ') traces in $OUTDIR/"
