# Data

Generated request traces (JSONL) for the experiments. Traces are produced by the
[workload generator](../tools/workload_generator/) from configs in
[`../configs/`](../configs/). This folder is the "how to make the data" story.

Configs are **inputs** (live in `configs/`, versioned). Traces are **outputs**
(land in `data/<dataset>/traces/`, git-ignored — they run to hundreds of MB).

```
data/
├── gen_gold_dataset.sh   # example sweep: builds the whole gold dataset
└── gold/traces/          # generated .jsonl traces (git-ignored)
```

## Setup

```bash
pip install -r tools/workload_generator/requirements.txt
```

A Gemma tokenizer is bundled at
`tools/utils/gemma_tokenizer/` and is the default, so
generation works out of the box. To use a different tokenizer, set
`tokenizer_path` in the config or pass `--tokenizer-path /path/to/tokenizer`.

## Three ways to generate

Run from the repo root. `TOOL` below is `tools/workload_generator`.

**1. One trace from one config**

```bash
python tools/workload_generator/mixed_workload_gen_ai_perf.py \
    --config configs/gold_config/s70_m20_a10.yaml \
    --out    data/gold/traces/s70_m20_a10.jsonl
```

Add `--hours 0.005` for a quick, small trace while iterating.

**2. Make a new mix from the base (by numbers)**

Derive a config from the base by setting the three fractions (they must sum to
1.0), then generate its trace:

```bash
python tools/workload_generator/write_mix_config.py \
    --base configs/gold_config/base.yaml --out configs/gold_config/s60_m40_a0.yaml \
    --single 0.6 --multi 0.4 --agentic 0.0

python tools/workload_generator/mixed_workload_gen_ai_perf.py \
    --config configs/gold_config/s60_m40_a0.yaml \
    --out    data/gold/traces/s60_m40_a0.jsonl
```

**3. The full gold sweep**

One command does step 2 for all 21 points of the ternary simplex and generates
every trace:

```bash
./data/gen_gold_dataset.sh
```

Configs are (re)written to `configs/gold_config/`, traces to `data/gold_traces/`.
Override the interpreter with `PYTHON=/path/to/python`, or the seed with `SEED=1`.
To build your own sweep, copy the script and edit `BASE` and the `COMBOS` list.
