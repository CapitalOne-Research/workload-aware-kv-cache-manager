#!/usr/bin/env python3
"""Build experiment-group batch specs from the ablation traces.

Each group file is a copy of a template batch spec (default
`runner/experiment_groups/my_batch.json`) with `input_workloads` (and, in
algo-ablation mode, `algos`) replaced — every other knob (base_config, vol_name,
wandb_key, commit, cuda_visible_devices, name_prefix, policy_extra_config,
publish) carries over verbatim. Edit the template and re-run to update them all.

GROUPING MODES (--group-by)
  algo-ablation  (default)  one group per (policy x ablation): every
                            input_lengths trace under `lru` is one group, every
                            input_lengths trace under `arc` is the next, and so
                            on. Group count = len(algos) x #ablations, and each
                            group runs ONE policy, so a group's runs all share a
                            vLLM build and vary only the knob being ablated.
  even                      the traces split round-robin into --groups buckets,
                            each running ALL algos. Even trace counts and an even
                            ablation mix per group.

The shared path prefix lives in the spec's `data_dir` (default
`data/ablations/traces`) and `input_workloads` hold only the remainder, e.g.
`input_lengths_s0_m50_a50/agentic_in1024.jsonl`. The runner rejoins them as
`f"{DATA_DIR}/{path.lstrip('/')}"`, with DATA_DIR resolved against the repo root
(generate_and_run.py). This also keeps the prefix out of the run-folder name,
which `_run_name()` builds by joining the trace path's segments with "_".
`data_dir` and `algos` are the template fields these specs intentionally change.

The ablation of a trace is read from its folder, supporting both the flattened
layout (`<ablation>_<mix>/<label>.jsonl`) and the nested one
(`<ablation>/<mix>/<label>.jsonl`).

USAGE   (run from the repo root)
    ./runner/make_ablation_groups.py                       # one group per algo x ablation
    ./runner/make_ablation_groups.py --dry-run             # show the split, write nothing
    ./runner/make_ablation_groups.py --group-by even --groups 8
    ./runner/make_ablation_groups.py --only-algos lru      # just the lru groups
    ./runner/make_ablation_groups.py --name-prefix "{algo}_{ablation}"  # per-group prefix

Then launch one group with:
    ./runner/run_experiments.sh -j runner/experiment_groups/lru_input_lengths.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# trailing turn-mix segment of a trace folder, e.g. "_s50_m50_a0", "_s33_m33_a33"
_MIX_SUFFIX = re.compile(r"_s\d+_m\d+_a\d+$")


def _ablation_of(trace: str, traces_dir: str) -> str:
    """Ablation name from a trace path, for either the flattened or nested layout."""
    parts = os.path.relpath(trace, traces_dir).split(os.sep)
    if len(parts) >= 3:                     # <ablation>/<mix>/<label>.jsonl
        return parts[0]
    if len(parts) == 2:                     # <ablation>_<mix>/<label>.jsonl
        return _MIX_SUFFIX.sub("", parts[0])
    return "unknown"                        # trace sitting loose in the root


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build experiment-group batch specs from the ablation traces.")
    p.add_argument("--group-by", choices=("algo-ablation", "even"), default="algo-ablation",
                   help="one group per policy x ablation (default), or an even "
                        "round-robin split across --groups buckets")
    p.add_argument("--groups", type=int, default=8,
                   help="bucket count for --group-by even (ignored otherwise)")
    p.add_argument("--only-algos", default="",
                   help="comma-separated subset of the template's algos to emit groups for")
    p.add_argument("--only-ablations", default="",
                   help="comma-separated subset of ablations to emit groups for")
    p.add_argument("--traces", default="data/ablations/traces",
                   help="folder to scan for .jsonl traces (default: data/ablations/traces)")
    p.add_argument("--data-dir", default=None,
                   help="value for each spec's data_dir; input_workloads are written "
                        "relative to it (default: the --traces folder)")
    p.add_argument("--template", default="runner/my_batch.json",
                   help="batch spec to copy every non-trace field from")
    p.add_argument("--out-dir", default="runner/experiment_groups",
                   help="where the group specs are written")
    p.add_argument("--prefix", default="ablation_group",
                   help="basename prefix, --group-by even only")
    p.add_argument("--name-prefix", default=None,
                   help="name_prefix for every run folder (default: ablations). "
                        "Placeholders: {algo}, {ablation}, {stem} (the group's file "
                        "stem), {g} (group number). Pass \"\" for none")
    p.add_argument("--name-prefix-from-template", action="store_true",
                   help="keep the template's name_prefix instead of --name-prefix")
    p.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    a = p.parse_args()

    if a.groups < 1:
        sys.exit("--groups must be >= 1")

    traces_dir = os.path.join(REPO_ROOT, a.traces)
    # The spec's data_dir carries the shared prefix and input_workloads carry only
    # the remainder; the runner rejoins them as f"{DATA_DIR}/{path}" with DATA_DIR
    # resolved against the repo root (generate_and_run.py). Keeping the prefix here
    # also keeps it out of _run_name(), which joins the trace path into the run
    # folder name.
    data_dir_rel = a.data_dir if a.data_dir is not None else a.traces
    path_base = os.path.join(REPO_ROOT, data_dir_rel)
    out_dir = os.path.join(REPO_ROOT, a.out_dir)
    with open(os.path.join(REPO_ROOT, a.template), encoding="utf-8") as f:
        spec = json.load(f)

    traces = sorted(
        os.path.join(root, fn)
        for root, _, files in os.walk(traces_dir)
        for fn in files if fn.endswith(".jsonl")
    )
    if not traces:
        sys.exit(f"no .jsonl traces found under {traces_dir}")

    algos = spec.get("algos", [])
    if a.only_algos:
        want = [s.strip() for s in a.only_algos.split(",") if s.strip()]
        unknown = [s for s in want if s not in algos]
        if unknown:
            sys.exit(f"--only-algos: not in the template's algos: {unknown}")
        algos = want

    # groups: (file stem, algos, [trace paths], placeholder values for name_prefix)
    groups: list[tuple[str, list[str], list[str], dict[str, str]]] = []
    if a.group_by == "algo-ablation":
        by_abl: dict[str, list[str]] = collections.OrderedDict()
        for t in traces:
            by_abl.setdefault(_ablation_of(t, traces_dir), []).append(t)
        ablations = sorted(by_abl)
        if a.only_ablations:
            want = [s.strip() for s in a.only_ablations.split(",") if s.strip()]
            unknown = [s for s in want if s not in by_abl]
            if unknown:
                sys.exit(f"--only-ablations: no traces for {unknown}; have {ablations}")
            ablations = want
        for algo in algos:
            for abl in ablations:
                groups.append((f"{algo}_{abl}", [algo], by_abl[abl],
                               {"algo": algo, "ablation": abl}))
    else:
        buckets: list[list[str]] = [[] for _ in range(a.groups)]
        for i, t in enumerate(traces):                   # round-robin: even mix
            buckets[i % a.groups].append(t)
        groups = [(f"{a.prefix}_{g:02d}", algos, b,
                    {"algo": "-".join(algos), "ablation": "mixed"})
                   for g, b in enumerate(buckets, 1)]

    name_prefix = "ablations" if a.name_prefix is None else a.name_prefix

    print(f"{len(traces)} traces, {len(algos)} algos -> {len(groups)} groups "
          f"(--group-by {a.group_by})")
    prefix_note = ("template's " + repr(spec.get("name_prefix", ""))
                   if a.name_prefix_from_template else repr(name_prefix))
    print(f"template: {a.template}   name_prefix: {prefix_note}\n")

    width = max((len(n) for n, _, _, _ in groups), default=20) + 6
    print(f"{'group':<{width}} {'name_prefix':<26} {'traces':>6} {'runs':>6} {'bytes':>10}")
    total_runs = 0
    for g, (stem, g_algos, g_traces, ph) in enumerate(groups, start=1):
        group_spec = dict(spec)                       # copy every knob verbatim
        group_spec["algos"] = g_algos                 # ...one policy per group here
        group_spec["input_workloads"] = [os.path.relpath(t, path_base) for t in g_traces]
        group_spec["data_dir"] = data_dir_rel         # ...the shared prefix lives here
        this_prefix = name_prefix.format(g=g, stem=stem, **ph)
        if not a.name_prefix_from_template:
            group_spec["name_prefix"] = this_prefix
        else:
            this_prefix = spec.get("name_prefix", "")
        if not a.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(group_spec, f, indent=2)
                f.write("\n")
        runs = len(g_traces) * len(g_algos)
        total_runs += runs
        nbytes = sum(os.path.getsize(t) for t in g_traces)
        print(f"{stem + '.json':<{width}} {this_prefix:<26} {len(g_traces):>6} "
              f"{runs:>6} {nbytes / 1024**3:>9.2f}G")

    print(f"\n{'TOTAL':<{width}} {'':<26} {sum(len(t) for _, _, t, _ in groups):>6} {total_runs:>6}")
    print("\n(dry run — nothing written)" if a.dry_run
          else f"\nwrote {len(groups)} specs to {a.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
