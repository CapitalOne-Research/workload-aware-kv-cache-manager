#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# runner/run_experiments.sh  —  Generate per-experiment run YAMLs and trigger them
#
# Thin wrapper around tools/perf_test_runner/generate_and_run.py.
# Every batch knob can be set here (or via the matching WAKMAN_* env var);
# anything unrecognized is forwarded verbatim to generate_and_run.py.
#
# Batch knobs (repeatable options build a list):
#   -j, --json SPEC        JSON file OR inline string with the batch spec. Every
#                          knob below can live in it, e.g.:
#                            {"algos": ["lru","s3_fifo"],
#                             "input_workloads": ["gold/traces/x.jsonl"],
#                             "base_config": "runner/base_perf_test_configs/a6000.yaml",
#                             "data_dir": "data", "vol_name": "<datavol>",
#                             "wandb_key": "<key>", "commit": "<sha>",
#                             "cuda_visible_devices": "0,1,2,3",
#                             "name_prefix": "<prefix>",
#                             "policy_extra_config": {"dynamic": {"start_policy": "lru"}},
#                             "publish": "<commit message>"}
#                          → passed through as WAKMAN_BATCH_JSON; generate_and_run.py
#                            parses it. Lists combine with -a/-f; every scalar field
#                            is overridden by its matching CLI flag below.
#   -f, --file FILE        input trace (repeatable)        → WAKMAN_INPUT_FILES
#   -a, --algo ALGO        eviction policy (repeatable)     → WAKMAN_ALGOS
#       --wandb-key KEY    W&B key for the batch            → WAKMAN_WANDB_KEY
#       --vol-name NAME    data volume (setup.vol_name)     → WAKMAN_VOL_NAME
#       --commit SHA       vLLM commit to build             → WAKMAN_VLLM_COMMIT
#       --name-prefix STR  experiment name prefix           → WAKMAN_NAME_PREFIX
#       --cuda-visible-devices IDS  GPU list for vLLM        → WAKMAN_CUDA_VISIBLE_DEVICES
#                                   (vllm.cuda_visible_devices, e.g. "0,1,2,3")
#       --base-config PATH base config to derive from        → WAKMAN_BASE_CONFIG
#                                   (absolute, or relative to the repo root)
#       --data-dir DIR     folder holding input traces        → WAKMAN_DATA_DIR
#                                   (default: repo data/; -f paths are relative to it)
#       --policy-extra-config JSON  per-policy extra config    → WAKMAN_POLICY_EXTRA_CONFIG
#                                   e.g. '{"dynamic":{"start_policy":"lru"}}'
#       --publish MSG      commit message → WAKMAN_PUBLISH. On a successful REAL
#                          batch (never --dry-run/--generate-only) generate_and_run.py
#                          git-adds runner/ experiments/ configs/, commits, and pushes.
# Any knob not set here falls back to the hardcoded default in generate_and_run.py.
#
# Forwarded flags:
#       --generate-only    only write the YAMLs, don't run
#       --dry-run          generate + run.py --dry-run (validate only)
#       --keep-going       don't stop the batch on failure
#
# Usage:
#   ./run_experiments.sh                                       # all defaults
#   ./run_experiments.sh -j batch.json                         # spec from a JSON file
#   ./run_experiments.sh -j '{"algos":["lru","s3-fifo"],"input_workloads":["/t.jsonl"]}'
#   ./run_experiments.sh -f trace_a.jsonl -f trace_b.jsonl     # two input traces
#   ./run_experiments.sh -a lru -a s3_fifo --generate-only     # two policies, YAMLs only
#   ./run_experiments.sh -f trace_a.jsonl --commit abc123 --dry-run
#
# Optional override:
#   PYTHON  — python interpreter to use (default: python3)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Resolve paths relative to this script so it works from any working directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="$HERE/../tools/perf_test_runner/generate_and_run.py"

if [ ! -f "$GEN_SCRIPT" ]; then
    echo "[run_experiments.sh] generate_and_run.py not found at $GEN_SCRIPT" >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"

# Collect batch knobs into env vars; forward everything else to generate_and_run.py.
INPUT_FILES=()
ALGOS=()
PASS_ARGS=()
need_arg() {   # need_arg <flag>: fail if the current option has no following value
    [ "$2" -gt 0 ] || { echo "[run_experiments.sh] $1 requires a value" >&2; exit 2; }
}
while [ $# -gt 0 ]; do
    case "$1" in
        -j|--json)        need_arg "$1" $(($# - 1)); shift; export WAKMAN_BATCH_JSON="$1" ;;
        -f|--file)        need_arg "$1" $(($# - 1)); shift; INPUT_FILES+=("$1") ;;
        -a|--algo)        need_arg "$1" $(($# - 1)); shift; ALGOS+=("$1") ;;
        --wandb-key)      need_arg "$1" $(($# - 1)); shift; export WAKMAN_WANDB_KEY="$1" ;;
        --vol-name)       need_arg "$1" $(($# - 1)); shift; export WAKMAN_VOL_NAME="$1" ;;
        --commit)         need_arg "$1" $(($# - 1)); shift; export WAKMAN_VLLM_COMMIT="$1" ;;
        --name-prefix)    need_arg "$1" $(($# - 1)); shift; export WAKMAN_NAME_PREFIX="$1" ;;
        --cuda-visible-devices)
                          need_arg "$1" $(($# - 1)); shift; export WAKMAN_CUDA_VISIBLE_DEVICES="$1" ;;
        --base-config)    need_arg "$1" $(($# - 1)); shift; export WAKMAN_BASE_CONFIG="$1" ;;
        --data-dir)       need_arg "$1" $(($# - 1)); shift; export WAKMAN_DATA_DIR="$1" ;;
        --policy-extra-config)
                          need_arg "$1" $(($# - 1)); shift; export WAKMAN_POLICY_EXTRA_CONFIG="$1" ;;
        --publish)        need_arg "$1" $(($# - 1)); shift; export WAKMAN_PUBLISH="$1" ;;
        *)                PASS_ARGS+=("$1") ;;
    esac
    shift
done

# Repeatable lists are handed over one item per line (generate_and_run.py splits
# on newlines/commas). Only export when set so a pre-existing env var survives.
if [ ${#INPUT_FILES[@]} -gt 0 ]; then
    printf -v WAKMAN_INPUT_FILES '%s\n' "${INPUT_FILES[@]}"
    export WAKMAN_INPUT_FILES
    echo "[run_experiments.sh] input files: ${INPUT_FILES[*]}"
fi
if [ ${#ALGOS[@]} -gt 0 ]; then
    printf -v WAKMAN_ALGOS '%s\n' "${ALGOS[@]}"
    export WAKMAN_ALGOS
    echo "[run_experiments.sh] algos: ${ALGOS[*]}"
fi

echo "[run_experiments.sh] $PYTHON $GEN_SCRIPT ${PASS_ARGS[*]:-}"
# generate_and_run.py does the work — including --publish/publish (git add/commit/
# push on a successful real batch), so its exit code is the script's.
exec "$PYTHON" "$GEN_SCRIPT" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
