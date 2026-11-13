# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process controller for the dynamic KV-cache eviction policy.

Runs a daemon thread inside the engine-core process (started by the
``Scheduler`` when the ``dynamic`` policy is active). Every interval it:

  1. drains the shadow trace in-process (completed-request block hashes + the
     window's average free-block occupancy),
  2. replays it through the bundled ``kvsim`` under LRU vs. S3-FIFO at that
     free-block capacity (kvsim runs as an isolated subprocess, so it is never
     imported into the serving process),
  3. applies hysteresis and, on a decision, *requests* a policy switch.

Thread-safety: the controller never mutates the block pool. It only records a
pending policy via ``switch_fn``; the scheduler applies it at the top of its
next step (single-threaded, between steps), so the switch cannot race an
insert/evict. This replaces the external HTTP controller with an auto-launched,
self-contained loop — no separate process, no start-time coordination, and the
shadow-trace buffer stays one-window small because it is always drained.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# (vLLM policy name, kvsim --policy name). vLLM's "s3_fifo" is small-free-first
# (no size target), which kvsim implements as "s3fifo_free" (POL_S3FIFO_FREE),
# NOT the classic 10%-target "s3fifo".
POLICIES: list[tuple[str, str]] = [("lru", "lru"), ("s3_fifo", "s3fifo_free")]

# Bundled kvsim run.py: <repo root>/tools/kvsim/run.py. parents[4] of
# vllm/v1/core/sched/dynamic_controller.py is the repo root.
DEFAULT_KVSIM_RUN = str(
    Path(__file__).resolve().parents[4] / "tools" / "kvsim" / "run.py"
)


class Hysteresis:
    """Switch only when a challenger beats the incumbent by ``margin`` for
    ``consecutive`` windows in a row."""

    def __init__(self, margin: float, consecutive: int) -> None:
        self._margin = margin
        self._consecutive = consecutive
        self._wins: dict[str, int] = {name: 0 for name, _ in POLICIES}

    def decide(self, incumbent: str, scores: dict[str, float]) -> str | None:
        best = max(scores, key=lambda p: scores[p])
        beats = scores[best] - scores.get(incumbent, 0.0)
        if best == incumbent or beats < self._margin:
            for name in self._wins:
                self._wins[name] = 0
            return None
        for name in self._wins:
            self._wins[name] = self._wins[name] + 1 if name == best else 0
        if self._wins[best] >= self._consecutive:
            self._wins[best] = 0
            return best
        return None


def write_mooncake_trace(payload: dict, path: Path) -> int:
    """Write the drained trace as a kvsim mooncake-format JSONL (arrival order).

    Prompt block hashes become ``hash_ids`` (the shareable prefix); generated
    blocks contribute residency pressure via ``output_length``. Returns the
    number of records written.
    """
    block_size = int(payload["block_size"])
    requests = sorted(payload["requests"], key=lambda r: r["arrival_time"])
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for req in requests:
            hashes = req["block_hashes"]
            n_prompt = max(0, min(int(req["num_prompt_blocks"]), len(hashes)))
            record = {
                "timestamp": req["arrival_time"],
                "input_length": n_prompt * block_size,
                "output_length": (len(hashes) - n_prompt) * block_size,
                "hash_ids": hashes[:n_prompt],
            }
            fh.write(json.dumps(record))
            fh.write("\n")
            written += 1
    return written


def run_kvsim(
    python_exe: str,
    kvsim_run: str,
    trace_path: Path,
    kvsim_policy: str,
    capacity_blocks: int,
    block_size: int,
    model: str,
    timeout: float,
) -> float:
    """Run one kvsim replay and return its overall hit rate."""
    cmd = [
        python_exe,
        kvsim_run,
        "--trace",
        str(trace_path),
        "--trace-format",
        f"mooncake:block_size={block_size}",
        "--policy",
        kvsim_policy,
        "--capacity-blocks",
        str(capacity_blocks),
        "--block-size",
        str(block_size),
        "--model",
        model,
        "--format",
        "json",
        "--output",
        "-",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kvsim ({kvsim_policy}) failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}"
        )
    return float(json.loads(proc.stdout)["summary"]["hit_rate"])


class DynamicEvictionController:
    """Daemon-thread control loop for the dynamic eviction policy.

    Args:
        drain_fn: returns the drained shadow-trace payload (and rolls it).
        incumbent_fn: returns the currently-active policy name.
        switch_fn: records a requested policy switch (applied by the scheduler).
        model: kvsim ``--model`` path (Gemma config → group layout).
        kvsim_run: path to kvsim ``run.py`` (default: the bundled copy).
    """

    def __init__(
        self,
        *,
        drain_fn: Callable[[], dict],
        incumbent_fn: Callable[[], str],
        switch_fn: Callable[[str], None],
        model: str,
        kvsim_run: str = DEFAULT_KVSIM_RUN,
        python_exe: str = sys.executable,
        interval_min: float = 30.0,
        margin: float = 0.02,
        consecutive: int = 2,
        min_requests: int = 50,
        dry_run: bool = False,
        kvsim_timeout: float = 600.0,
    ) -> None:
        self._drain_fn = drain_fn
        self._incumbent_fn = incumbent_fn
        self._switch_fn = switch_fn
        self._model = model
        self._kvsim_run = kvsim_run
        self._python_exe = python_exe
        self._interval_s = max(1.0, interval_min * 60.0)
        self._min_requests = min_requests
        self._dry_run = dry_run
        self._kvsim_timeout = kvsim_timeout
        self._hysteresis = Hysteresis(margin, consecutive)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._model:
            logger.warning(
                "dynamic eviction controller disabled: no model path for kvsim"
            )
            return
        if not Path(self._kvsim_run).exists():
            logger.warning(
                "dynamic eviction controller disabled: kvsim not found at %s",
                self._kvsim_run,
            )
            return
        self._thread = threading.Thread(
            target=self._loop, name="dynamic-eviction-controller", daemon=True
        )
        self._thread.start()
        logger.info(
            "dynamic eviction controller started (interval=%.1fs, model=%s, "
            "dry_run=%s)",
            self._interval_s,
            self._model,
            self._dry_run,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Wait a full interval before the first cycle so a window has data.
        while not self._stop.wait(self._interval_s):
            try:
                self._run_cycle()
            except Exception as e:  # never let the daemon thread die
                logger.warning("dynamic eviction cycle failed: %s", e)

    def _run_cycle(self) -> None:
        payload = self._drain_fn()
        n_requests = int(payload.get("num_requests", 0))
        incumbent = self._incumbent_fn() or "lru"
        capacity = max(1, round(float(payload.get("avg_free_blocks", 0.0))))
        if n_requests < self._min_requests:
            logger.info(
                "dynamic eviction: %d requests (< %d), staying on %s",
                n_requests,
                self._min_requests,
                incumbent,
            )
            return
        block_size = int(payload["block_size"])
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "shadow_trace.jsonl"
            if write_mooncake_trace(payload, trace_path) == 0:
                return
            scores = {
                vllm_name: run_kvsim(
                    self._python_exe,
                    self._kvsim_run,
                    trace_path,
                    kvsim_name,
                    capacity,
                    block_size,
                    self._model,
                    self._kvsim_timeout,
                )
                for vllm_name, kvsim_name in POLICIES
            }
        logger.info(
            "dynamic eviction hit rates @ %d blocks: %s (incumbent=%s)",
            capacity,
            {k: round(v, 4) for k, v in scores.items()},
            incumbent,
        )
        winner = self._hysteresis.decide(incumbent, scores)
        if winner is None:
            return
        if self._dry_run:
            logger.info(
                "dynamic eviction: would switch %s -> %s (dry-run)",
                incumbent,
                winner,
            )
            return
        self._switch_fn(winner)
        logger.info("dynamic eviction: requested switch %s -> %s", incumbent, winner)
