# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow-trace tap for the dynamic KV-cache eviction policy.

When the ``dynamic`` GPU eviction policy is active, an external controller
periodically decides whether LRU or S3-FIFO would serve the current workload
better and switches the policy at runtime. To make that decision it replays a
recent trace of completed requests through an offline simulator, so the engine
must hand it that trace.

This tap is a small, dependency-light ring buffer owned by the ``Scheduler``.
At request completion it records the request's block-hash sequence and arrival
time; every scheduler step it samples free-block occupancy into a running
average (the "effective capacity" the prefix cache actually gets once
concurrent requests pin part of the pool). ``drain`` hands the accumulated
window to the controller and rolls it. It is deliberately free of torch / vLLM
engine imports so it can be unit-tested in isolation.
"""

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any


def _hash_to_id(block_hash: Any) -> int:
    """Map a block hash to a stable, JSON-safe opaque id.

    ``BlockHash`` is ``bytes``; the first 8 bytes of an already-uniform hash
    give a stable 64-bit id (collision-negligible within one window). Ints
    (test stubs) pass through.
    """
    if isinstance(block_hash, (bytes, bytearray)):
        return int.from_bytes(bytes(block_hash[:8]), "big")
    return int(block_hash)


@dataclass
class _Entry:
    arrival_time: float
    block_hashes: tuple[int, ...]
    num_prompt_blocks: int


class ShadowTraceTap:
    """Bounded shadow trace of completed requests + a free-block running average.

    Args:
        num_gpu_blocks: Total GPU block-pool capacity (for the drained payload).
        block_size: Hash block size, used to split a request's block-hash
            sequence into prompt vs. generated blocks.
        max_requests: Hard cap on buffered requests; oldest are dropped first so
            a long window at high QPS cannot grow without bound.
    """

    def __init__(
        self, num_gpu_blocks: int, block_size: int, max_requests: int = 200_000
    ) -> None:
        self._num_gpu_blocks = num_gpu_blocks
        self._block_size = block_size
        self._buf: deque[_Entry] = deque(maxlen=max_requests)
        self._free_sum = 0.0
        self._free_samples = 0
        # record/sample run on the scheduler thread; drain runs on the
        # in-process controller thread — guard the shared state.
        self._lock = Lock()

    def record_finished(
        self,
        arrival_time: float,
        block_hashes: Iterable[Any],
        num_prompt_tokens: int,
    ) -> None:
        """Record a completed request's block-hash trace.

        Called at the top of ``Scheduler._free_request`` while the request is
        still intact. ``block_hashes`` is the request's full (prompt + output)
        block-hash sequence; the prompt prefix is what the simulator measures
        cache hits against.
        """
        ids = tuple(_hash_to_id(h) for h in block_hashes)
        if self._block_size > 0:
            num_prompt_blocks = min(num_prompt_tokens // self._block_size, len(ids))
        else:
            num_prompt_blocks = 0
        with self._lock:
            self._buf.append(_Entry(arrival_time, ids, num_prompt_blocks))

    def sample_free_blocks(self, num_free_blocks: int) -> None:
        """Add one free-block occupancy sample (once per scheduler step)."""
        with self._lock:
            self._free_sum += num_free_blocks
            self._free_samples += 1

    def drain(self) -> dict[str, Any]:
        """Return the accumulated window and roll it (buffer + average reset).

        The payload is strictly primitive (dict / list / int / float) so it
        rides the existing utility-RPC path without custom serialization.
        """
        with self._lock:
            entries = list(self._buf)
            free_sum = self._free_sum
            free_samples = self._free_samples
            self._buf.clear()
            self._free_sum = 0.0
            self._free_samples = 0
        if free_samples:
            avg_free_blocks = free_sum / free_samples
        else:
            avg_free_blocks = float(self._num_gpu_blocks)
        payload: dict[str, Any] = {
            "num_gpu_blocks": self._num_gpu_blocks,
            "block_size": self._block_size,
            "avg_free_blocks": avg_free_blocks,
            "num_requests": len(entries),
            "requests": [
                {
                    "arrival_time": e.arrival_time,
                    "block_hashes": list(e.block_hashes),
                    "num_prompt_blocks": e.num_prompt_blocks,
                }
                for e in entries
            ],
        }
        return payload

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# Kept for type clarity at call sites that pass a slice of hashes.
BlockHashSeq = Sequence[Any]
