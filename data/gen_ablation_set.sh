#!/usr/bin/env bash
# =============================================================================
# gen_ablation_set.sh — build the ABLATION dataset (one knob at a time).
#
# The sweep is described by configs/ablation_list/ablations.json. For every base mix (base_fractions_*),
# for every workload profile present in that mix, for every value of an ablation,
# it emits ONE config that differs from configs/gold_config/base.yaml in exactly
# one knob (plus the profile fractions that define the mix), and generates that
# config's trace.
#
#   configs land in  configs/ablation_configs/<ablation>/s<S>_m<M>_a<A>/<profile>_<tag><value>.yaml
#   traces  land in  data/ablations/traces/<ablation>/s<S>_m<M>_a<A>/<profile>_<tag><value>.jsonl
#
# e.g. configs/ablation_configs/input_lengths/s50_m50_a0/multi_turn_in8192.yaml
#
# USAGE   (run from anywhere)
#   ./data/gen_ablation_set.sh
# ENV (optional)
#   PYTHON        interpreter with numpy + tokenizers   (default: python3)
#   SEED          empty = use the base config's seed; set an int to override
#   HOURS         short duration override for quick iteration (e.g. 0.005)
#   ABLATIONS     comma-separated subset, e.g. "input_lengths,turns"
#                 (default: all — input_lengths, output_lengths, num_services,
#                  turns, inter_turn_time)
#   JOBS          configs to generate in parallel (default: half the cores).
#                 Each config is independent — its own config in, its own trace
#                 out — so they parallelize cleanly. The limit is RAM, not CPU:
#                 the generator buffers every record in memory before writing,
#                 so peak RSS tracks trace size (~3 GB for a 131072-token input
#                 sweep point). Half the cores keeps the big ones off swap;
#                 raise it when the sweep is all small traces. JOBS=1 restores
#                 the old serial behaviour (and live per-config output).
#   CONFIGS_ONLY  1 = write the YAML configs, skip trace generation
#   DRY_RUN       1 = just print the plan (nothing is written)
#   ABL_JSON      the sweep values   (default: configs/ablation_list/ablations.json)
#   BASE          base config        (default: configs/gold_config/base.yaml)
#
# The two Python tools it drives live next to this script, in
# data/ablation_workload_config_generator/ (see its README).
# =============================================================================
set -euo pipefail

# absolute path to this script — the parallel dispatcher re-invokes it as a
# worker, and $0 alone would break once we cd to the repo root.
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# run from the repo root: the configs' `tokenizer_path` is repo-relative, so the
# trace generator only resolves the bundled tokenizer from here.
cd "$REPO_ROOT"
GEN="$REPO_ROOT/tools/workload_generator"                      # the trace generator
TOOL="$REPO_ROOT/tools/ablation_workload_config_generator"     # plan + config writer
BASE="${BASE:-$REPO_ROOT/configs/gold_config/base.yaml}"
ABL_JSON="${ABL_JSON:-$REPO_ROOT/configs/ablation_configs/ablations.json}"
CFGDIR="$REPO_ROOT/configs/ablation_configs"
OUTDIR="$REPO_ROOT/data/ablations/traces"

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-}"
HOURS="${HOURS:-}"
ABLATIONS="${ABLATIONS:-}"
CONFIGS_ONLY="${CONFIGS_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

_ncpu="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)"
JOBS="${JOBS:-$((_ncpu / 4))}"
[ "$JOBS" -ge 1 ] || JOBS=1

# ── worker ───────────────────────────────────────────────────────────────────
# One plan row -> one config + one trace. Re-invoked per row by the dispatcher
# below (everything above is recomputed identically; SEED/HOURS/BASE/... arrive
# through the environment). Kept in this file so there is one source of truth
# for how a row becomes a trace.
if [ "${1:-}" = "--worker" ]; then
  IFS=$'\t' read -r abl mix s m a profile value label <<<"${2:-}"
  [ -n "${abl:-}" ] || exit 0

  cfg="$CFGDIR/$abl/$mix/${label}.yaml"
  out="$OUTDIR/${abl}_${mix}/${label}.jsonl"
  mkdir -p "$(dirname "$cfg")" "$(dirname "$out")"   # mkdir -p is idempotent, safe to race

  # percent -> fraction (kept exact-ish so the sum-to-1.0 check passes)
  fs=$(awk "BEGIN{printf \"%.10g\", $s/100}")
  fm=$(awk "BEGIN{printf \"%.10g\", $m/100}")
  fa=$(awk "BEGIN{printf \"%.10g\", $a/100}")

  "$PYTHON" "$TOOL/write_ablation_config.py" \
      --base "$BASE" --out "$cfg" \
      --single "$fs" --multi "$fm" --agentic "$fa" \
      --ablation "$abl" --profile "$profile" --value "$value"

  if [ "$CONFIGS_ONLY" = "1" ]; then
    echo "config: $cfg"
    exit 0
  fi

  # Buffer the generator's output and emit it as one block on completion, so
  # concurrent workers can't interleave mid-line. With JOBS=1 this just means
  # the block prints when the config finishes instead of as it runs.
  cmd=("$PYTHON" "$GEN/mixed_workload_gen_ai_perf.py" --config "$cfg" --out "$out")
  [ -n "$SEED" ] && cmd+=(--seed "$SEED")
  [ -n "$HOURS" ] && cmd+=(--hours "$HOURS")
  if log=$("${cmd[@]}" 2>&1); then
    printf '=== %s  %s  %s=%s  ->  %s ===\n%s\n' "$abl" "$mix" "$profile" "$value" "$out" "$log"
  else
    printf '!!! FAILED  %s  %s  %s=%s  ->  %s\n%s\n' \
      "$abl" "$mix" "$profile" "$value" "$out" "$log" >&2
    exit 1
  fi
  exit 0
fi

# ── plan ─────────────────────────────────────────────────────────────────────
plan_args=(--ablations "$ABL_JSON")
[ -n "$ABLATIONS" ] && plan_args+=(--only "$ABLATIONS")

if [ "$DRY_RUN" = "1" ]; then
  "$PYTHON" "$TOOL/ablation_plan.py" "${plan_args[@]}" \
    | awk -F'\t' '{printf "%-16s %-14s %s = %s\n", $1, $2, $6, $7}'
  echo "plan: $("$PYTHON" "$TOOL/ablation_plan.py" "${plan_args[@]}" | wc -l | tr -d ' ') configs (dry run, nothing written)"
  exit 0
fi

mkdir -p "$CFGDIR" "$OUTDIR"

# plan rows: <ablation> <mix> <single%> <multi%> <agentic%> <profile> <value> <label>
# Every row is independent (unique config path, unique trace path, seed from the
# config), so the only ordering the output depends on is none.
rows="$("$PYTHON" "$TOOL/ablation_plan.py" "${plan_args[@]}" | grep .)"
n=$(printf '%s\n' "$rows" | wc -l | tr -d ' ')

echo "plan: $n configs, $JOBS at a time"
# NUL-delimited: BSD xargs -I collapses tabs to spaces, which would destroy the
# row's field boundaries, so feed records with -0 -n1 instead — that passes each
# row through as one argument with its tabs intact. A failing worker makes xargs
# exit non-zero, which set -e turns into an abort (jobs in flight still finish).
printf '%s\n' "$rows" | tr '\n' '\0' | xargs -0 -P "$JOBS" -n1 "$BASH" "$SELF" --worker

if [ "$CONFIGS_ONLY" = "1" ]; then
  echo "done: $n configs in $CFGDIR/ (traces skipped)"
else
  echo "done: $n configs in $CFGDIR/, $(find "$OUTDIR" -name '*.jsonl' | wc -l | tr -d ' ') traces in $OUTDIR/"
fi
