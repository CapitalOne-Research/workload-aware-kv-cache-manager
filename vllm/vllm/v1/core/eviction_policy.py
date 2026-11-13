# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU KV-cache block eviction policies.

The ``BlockPool`` delegates all free-block bookkeeping to a ``GPUCachePolicy``.
The policy owns whatever data structure fits its strategy (a recency-ordered
linked list for LRU, frequency buckets for LFU); ``BlockPool`` only speaks the
policy interface and never assumes a particular representation.
"""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)

# Upper bound on an LFU frequency counter. Counters saturate here so the
# frequency universe stays bounded, which is what keeps min-frequency lookup
# O(1). The value is large enough that saturation is practically irrelevant;
# a saturated block is simply "evicted last" rather than pinned forever.
DEFAULT_LFU_MAX_FREQ = 65535

# Fraction of the GPU block pool the two-queue ("2q") policy sizes its ghost
# (recently-evicted content) table to. 1 => ghost table holds up to one entry
# per pool block.
DEFAULT_GHOST_CAPACITY_FACTOR = 1


class GPUCachePolicy(ABC):
    """Interface for choosing which free KV-cache blocks to evict.

    A free block is an eviction candidate: it has ``ref_cnt == 0`` and either
    holds cached content (a block hash) worth keeping, or is worthless
    (hashless) and should be reused first. The policy decides the order in
    which candidates are handed back for reuse.
    """

    @abstractmethod
    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        """Register blocks that just became eviction candidates.

        Args:
            blocks: Newly freed blocks, ordered by eviction priority (the
                first block is the best candidate to evict).
        """
        raise NotImplementedError

    def insert(self, block: KVCacheBlock) -> None:
        """Register a single eviction candidate."""
        self.insert_n([block])

    @abstractmethod
    def remove(self, block: KVCacheBlock) -> None:
        """Drop a block from the candidate set (it is being reused).

        Must be a no-op if the block is not currently a candidate.
        """
        raise NotImplementedError

    @abstractmethod
    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        """Record a cache hit on a block (accounting only, no reordering).

        Args:
            block: The block that was hit.
            request_priority: Priority of the hitting request (lower = more
                important). Only the priority policy consumes it; recency- and
                frequency-based policies ignore it.
        """
        raise NotImplementedError

    def on_cache(self, block: KVCacheBlock, request_priority: int) -> None:
        """Record that ``block``'s content was (re)cached by a request.

        Called when a block first becomes cached (a cache miss that fills and
        caches fresh content), which — unlike a cache hit — never goes through
        ``touch``. Only the priority policy uses it, to learn the caching
        request's priority; other policies ignore it (default no-op).
        """

    @abstractmethod
    def evict_n(self, n: int) -> list[KVCacheBlock]:
        """Remove and return the ``n`` best eviction candidates."""
        raise NotImplementedError

    @abstractmethod
    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return all candidates in eviction order (best candidate first)."""
        raise NotImplementedError

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        """Yield free blocks in eviction order (best candidate first).

        ``start_after`` requests a resumable cursor: when the block is still a
        free candidate, iteration begins just after it, letting a caller sweep
        the free set incrementally across calls without rescanning from the
        front. When ``start_after`` is ``None`` or no longer free, iteration
        begins at the best candidate.

        The base implementation is correct but O(n) in the number of free
        blocks; policies with a traversable structure (LRU, LFU) override it to
        honor the cursor in O(1) per step.
        """
        blocks = self.get_all_free_blocks()
        start = 0
        if start_after is not None:
            for i, block in enumerate(blocks):
                if block is start_after:
                    start = i + 1
                    break
        yield from blocks[start:]

    def reset(self) -> None:
        """Forget any accumulated cache statistics (prefix-cache reset)."""

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        """Inform the policy of the total block-pool capacity.

        Called once by ``BlockPool`` right after construction. Capacity-aware
        policies (S3-FIFO sizes its ghost queue from it) override this; recency-,
        frequency-, and priority-based policies ignore it (default no-op).
        """

    def as_free_block_queue(self) -> FreeKVCacheBlockQueue:
        """Return the underlying LRU free-block queue, if the policy has one.

        This exists only for legacy call sites (the CPU-offloading connector,
        some tests) that reach into the recency-ordered linked list directly.
        Policies without such a structure raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a FreeKVCacheBlockQueue; "
            "call sites that walk the free list directly require the 'lru' "
            "eviction policy."
        )

    @abstractmethod
    def __len__(self) -> int:
        """Number of eviction candidates currently held."""
        raise NotImplementedError


class LRUGPUCachePolicy(GPUCachePolicy):
    """Least-recently-used eviction.

    Thin wrapper over ``FreeKVCacheBlockQueue``: the candidate at the front of
    the list is the least recently used and is evicted first. Worthless
    (hashless) blocks are pushed to the very front so they are reused before
    any cached block. Recency is positional, so ``touch`` is a no-op.
    """

    def __init__(self) -> None:
        self._queue = FreeKVCacheBlockQueue([])

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        without_hash = [b for b in blocks if b.block_hash is None]
        with_hash = [b for b in blocks if b.block_hash is not None]
        # Hashless blocks have no reuse value: evict them first (front).
        self._queue.prepend_n(without_hash)
        self._queue.append_n(with_hash)

    def remove(self, block: KVCacheBlock) -> None:
        # Honor the interface contract: a no-op if the block is not a candidate.
        # A linked block always has both neighbors (the fake head/tail
        # sentinels); a None link means it is not in the free list.
        if block.prev_free_block is None or block.next_free_block is None:
            return
        self._queue.remove(block)

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        pass

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        return self._queue.popleft_n(n)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return self._queue.get_all_free_blocks()

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        q = self._queue
        tail = q.fake_free_list_tail
        # Resume just after ``start_after`` only if it is still linked in this
        # queue (both neighbors set); otherwise restart from the front.
        if (
            start_after is not None
            and start_after.prev_free_block is not None
            and start_after.next_free_block is not None
        ):
            node = start_after.next_free_block
        else:
            node = q.fake_free_list_head.next_free_block
        while node is not None and node is not tail:
            nxt = node.next_free_block  # capture before the caller mutates node
            yield node
            node = nxt

    def as_free_block_queue(self) -> FreeKVCacheBlockQueue:
        return self._queue

    def __len__(self) -> int:
        return self._queue.num_free_blocks


class LFUGPUCachePolicy(GPUCachePolicy):
    """Least-frequently-used eviction with persistent (ghost) frequencies.

    Frequency is a property of the cached *content*, keyed by the block hash,
    not of the physical block. A count therefore survives eviction: when the
    same content is admitted again its previous frequency is resurrected and
    keeps climbing. This is a simplified TinyLFU (exact counts, no sketch).

    Layout (all operations O(1) w.r.t. the number of blocks and requests):

    - ``_freq``: ``content hash -> access count``. The ghost table. Values
      saturate at ``max_freq``; the number of entries is unbounded by default
      (set ``ghost_capacity`` to LRU-trim it for long-running servers). The
      table is ordered by recency of the last access/(re)insert, so trimming
      targets truly cold content. A resident-but-idle block's count can still
      be dropped under a tight bound, which is harmless: eviction never reads
      ``_freq``, so the block simply restarts at frequency 1 on its next hit.
    - ``_buckets``: ``bucket -> FreeKVCacheBlockQueue`` of free blocks at that
      frequency. Bucket 0 holds worthless (hashless) blocks and is drained
      first; buckets ``1..max_freq`` hold cached blocks by frequency. Each queue
      is the same intrusive O(1) doubly-linked list the LRU policy uses, so
      within a bucket the least-recently-freed block is evicted first (the LRU
      tie-break) with no per-operation allocation.
    - ``_occ``: an integer bitset marking non-empty buckets, so the lowest
      non-empty bucket is found via ``_occ & -_occ`` without scanning.

    Each block records its current bucket in ``KVCacheBlock.eviction_bucket``,
    so a block that gets a cache hit is removed from the correct frequency queue
    in O(1) (the bucket cannot be derived from ``_freq[hash]``: duplicate blocks
    can share a hash and the count changes on the very hit being processed).

    Caveats:
    - ``touch`` only updates the ghost count; it never moves a block between
      buckets. Callers must ``remove`` a block from the candidate set before
      touching it (``BlockPool`` does: a cache hit removes the block from the
      free pool before recording the hit).
    - The KV-connector force-eviction path (``BlockPool.evict_blocks``) is
      intentionally *not* routed here. A block whose content is invalidated
      while it is a candidate keeps its frequency bucket until it is
      reallocated. This matches the LRU policy (which also leaves such blocks
      in place) and is harmless: the block is still a valid candidate and the
      scheduler reuses it promptly.
    """

    def __init__(
        self,
        max_freq: int = DEFAULT_LFU_MAX_FREQ,
        ghost_capacity: int | None = None,
    ) -> None:
        assert max_freq >= 1
        assert ghost_capacity is None or ghost_capacity >= 1
        self._max_freq = max_freq
        self._ghost_capacity = ghost_capacity
        self._freq: OrderedDict[BlockHashWithGroupId, int] = OrderedDict()
        self._buckets: dict[int, FreeKVCacheBlockQueue] = {}
        self._occ = 0
        self._num_free = 0

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.insert(block)

    def insert(self, block: KVCacheBlock) -> None:
        block_hash = block.block_hash
        if block_hash is None:
            # Worthless: no cache value, evict before any cached block.
            bucket = 0
        else:
            freq = self._freq.get(block_hash)
            if freq is None:
                # Freshly cached content enters at frequency 1.
                freq = 1
            # Record on every (re)insert, not just fresh ones, so content that
            # is actively being cached keeps a recent slot in a bounded ghost
            # table and is not trimmed while still in use. For the unbounded
            # default this just re-writes the same value (a no-op).
            self._record_freq(block_hash, freq)
            bucket = freq
        queue = self._buckets.get(bucket)
        if queue is None:
            queue = self._buckets[bucket] = FreeKVCacheBlockQueue([])
            self._occ |= 1 << bucket
        queue.append(block)  # tail = most recently freed
        block.eviction_bucket = bucket
        self._num_free += 1

    def remove(self, block: KVCacheBlock) -> None:
        bucket = block.eviction_bucket
        if bucket == -1:
            # Not a candidate (already in use); nothing to do.
            return
        block.eviction_bucket = -1
        queue = self._buckets[bucket]
        queue.remove(block)  # O(1) via the block's own prev/next links
        if queue.num_free_blocks == 0:
            del self._buckets[bucket]
            self._occ &= ~(1 << bucket)
        self._num_free -= 1

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        block_hash = block.block_hash
        if block_hash is None:
            return
        new_freq = min(self._freq.get(block_hash, 0) + 1, self._max_freq)
        self._record_freq(block_hash, new_freq)

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        if n <= 0:
            return []
        if self._num_free < n:
            raise ValueError(
                f"Cannot evict {n} blocks; only {self._num_free} are free"
            )
        victims: list[KVCacheBlock] = []
        while len(victims) < n:
            bucket = (self._occ & -self._occ).bit_length() - 1
            queue = self._buckets[bucket]
            take = min(queue.num_free_blocks, n - len(victims))
            for block in queue.popleft_n(take):  # front = least recently freed
                block.eviction_bucket = -1
                victims.append(block)
            if queue.num_free_blocks == 0:
                del self._buckets[bucket]
                self._occ &= ~(1 << bucket)
        self._num_free -= n
        return victims

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        result: list[KVCacheBlock] = []
        for bucket in sorted(self._buckets):
            result.extend(self._buckets[bucket].get_all_free_blocks())
        return result

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        # Resume within start_after's bucket if it is still a free candidate;
        # otherwise sweep every bucket from the lowest (best candidate) upward.
        # buckets are >= 0, so the -1 sentinel disables skipping/resumption.
        resume_bucket = -1
        resume_node = None
        if start_after is not None and start_after.eviction_bucket != -1:
            resume_bucket = start_after.eviction_bucket
            resume_node = start_after.next_free_block
        for bucket in sorted(self._buckets):
            if bucket < resume_bucket:
                continue
            queue = self._buckets[bucket]
            if bucket == resume_bucket:
                node = resume_node
            else:
                node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            while node is not None and node is not tail:
                nxt = node.next_free_block  # capture before caller mutates node
                yield node
                node = nxt

    def reset(self) -> None:
        # Called after BlockPool clears every block hash, so all candidates are
        # now hashless. Drop popularity stats and re-home them into bucket 0.
        blocks = self.get_all_free_blocks()
        self._freq.clear()
        self._buckets.clear()
        self._occ = 0
        self._num_free = 0
        self.insert_n(blocks)

    def __len__(self) -> int:
        return self._num_free

    def _record_freq(self, block_hash: BlockHashWithGroupId, freq: int) -> None:
        self._freq[block_hash] = freq
        if self._ghost_capacity is not None:
            self._freq.move_to_end(block_hash)
            while len(self._freq) > self._ghost_capacity:
                self._freq.popitem(last=False)


# Marker stored in ``KVCacheBlock.eviction_bucket`` for a free block that lives
# in the priority policy's hashless queue (distinct from -1 = "not free" and
# from any real, non-negative priority key).
_HASHLESS_BUCKET = -2

# Bucket key for a cached free block whose priority was never recorded. It sorts
# as the least important cached content (evicted right after hashless blocks),
# so an unknown-priority block is never wrongly protected. In the real
# BlockPool flow every cached block gets a priority via ``on_cache``/``touch``,
# so this is a defensive fallback.
_UNKNOWN_PRIORITY = 1 << 62


class PriorityGPUCachePolicy(GPUCachePolicy):
    """Priority eviction: high-priority content is evicted last.

    Priority is a property of the *request*, not the content, so — unlike LFU's
    self-counted frequency — it is supplied from outside: each referencing
    request folds its priority into the block via ``touch`` (cache hits) and
    ``on_cache`` (fresh caches). vLLM's convention is **lower number = more
    important** (``request.priority``; default 0). A block therefore keeps the
    best (lowest) priority of any request that used it, so a block shared by a
    high- and a low-priority request is protected at the high-priority level.

    There is no ghost table and no persistence: importance lives on the block
    (``KVCacheBlock.eviction_priority``) and is forgotten when the block is
    reused for new content. This fits a small, fixed set of priority tiers.

    Layout:

    - ``_hashless``: a ``FreeKVCacheBlockQueue`` of worthless (hashless) blocks,
      drained before any cached block.
    - ``_buckets``: ``priority -> FreeKVCacheBlockQueue`` of cached free blocks
      at that priority. Keys are the (few) distinct priorities in play; each
      queue is the same intrusive O(1) LRU list used elsewhere, so within a
      priority the least-recently-freed block is evicted first.

    Eviction order (best candidate first): hashless, then buckets by descending
    priority number (least important first) down to priority 0 (evicted last).

    Each block records the queue it currently sits in via
    ``KVCacheBlock.eviction_bucket`` (a priority key, ``_HASHLESS_BUCKET``, or
    -1 when not free) so a cache hit removes it in O(1). Negative request
    priorities are clamped to 0 (cannot be more important than the top tier).

    Caveats (mirroring the LFU policy):

    - ``touch``/``on_cache`` only fold the priority onto the block; they never
      move a block between buckets. ``BlockPool`` removes a block from the
      candidate set before recording a hit.
    - The KV-connector force-eviction path (``BlockPool.evict_blocks``) is not
      routed here; a block whose content is invalidated while free keeps its
      bucket until it is reused, which is harmless.
    """

    def __init__(self) -> None:
        self._hashless = FreeKVCacheBlockQueue([])
        self._buckets: dict[int, FreeKVCacheBlockQueue] = {}
        self._num_free = 0

    def _fold_priority(self, block: KVCacheBlock, request_priority: int) -> None:
        # Keep the most important (lowest) priority seen; clamp negatives to 0.
        priority = max(request_priority, 0)
        current = block.eviction_priority
        block.eviction_priority = (
            priority if current < 0 else min(current, priority)
        )

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.insert(block)

    def insert(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            # Worthless: no cache value, evict before any cached block.
            self._hashless.append(block)
            block.eviction_bucket = _HASHLESS_BUCKET
        else:
            priority = block.eviction_priority
            key = priority if priority >= 0 else _UNKNOWN_PRIORITY
            queue = self._buckets.get(key)
            if queue is None:
                queue = self._buckets[key] = FreeKVCacheBlockQueue([])
            queue.append(block)  # tail = most recently freed
            block.eviction_bucket = key
        self._num_free += 1

    def remove(self, block: KVCacheBlock) -> None:
        bucket = block.eviction_bucket
        if bucket == -1:
            # Not a candidate (already in use); nothing to do.
            return
        block.eviction_bucket = -1
        # Keep eviction_priority: a cache hit removes the block, records the hit,
        # then re-inserts it, and the accumulated priority must survive that.
        if bucket == _HASHLESS_BUCKET:
            self._hashless.remove(block)
        else:
            queue = self._buckets[bucket]
            queue.remove(block)  # O(1) via the block's own prev/next links
            if queue.num_free_blocks == 0:
                del self._buckets[bucket]
        self._num_free -= 1

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        if block.block_hash is None:
            return
        self._fold_priority(block, request_priority)

    def on_cache(self, block: KVCacheBlock, request_priority: int) -> None:
        if block.block_hash is None:
            return
        self._fold_priority(block, request_priority)

    def _take_from(
        self, queue: FreeKVCacheBlockQueue, n: int, victims: list[KVCacheBlock]
    ) -> None:
        for block in queue.popleft_n(n):  # front = least recently freed
            block.eviction_bucket = -1
            block.eviction_priority = -1  # reused for new content; forget it
            victims.append(block)

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        if n <= 0:
            return []
        if self._num_free < n:
            raise ValueError(
                f"Cannot evict {n} blocks; only {self._num_free} are free"
            )
        victims: list[KVCacheBlock] = []
        # Hashless blocks first: no reuse value.
        if self._hashless.num_free_blocks:
            take = min(self._hashless.num_free_blocks, n)
            self._take_from(self._hashless, take, victims)
        # Then cached blocks, least important (highest priority number) first.
        while len(victims) < n:
            key = max(self._buckets)
            queue = self._buckets[key]
            take = min(queue.num_free_blocks, n - len(victims))
            self._take_from(queue, take, victims)
            if queue.num_free_blocks == 0:
                del self._buckets[key]
        self._num_free -= n
        return victims

    def _ordered_queues(self) -> list[tuple[int, FreeKVCacheBlockQueue]]:
        # Eviction order: hashless, then buckets by descending priority number.
        ordered: list[tuple[int, FreeKVCacheBlockQueue]] = [
            (_HASHLESS_BUCKET, self._hashless)
        ]
        ordered.extend(
            (key, self._buckets[key]) for key in sorted(self._buckets, reverse=True)
        )
        return ordered

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        result: list[KVCacheBlock] = []
        for _, queue in self._ordered_queues():
            result.extend(queue.get_all_free_blocks())
        return result

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        # Resume within start_after's queue if it is still a free candidate;
        # otherwise sweep every queue from the best candidate onward.
        resume_bucket: int | None = None
        resume_node: KVCacheBlock | None = None
        if start_after is not None and start_after.eviction_bucket != -1:
            resume_bucket = start_after.eviction_bucket
            resume_node = start_after.next_free_block
        started = resume_bucket is None
        for key, queue in self._ordered_queues():
            if not started:
                if key != resume_bucket:
                    continue
                started = True
                node = resume_node
            else:
                node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            while node is not None and node is not tail:
                nxt = node.next_free_block  # capture before caller mutates node
                yield node
                node = nxt

    def reset(self) -> None:
        # Called after BlockPool clears every block hash, so all candidates are
        # now hashless. Drop priorities and re-home them into the hashless queue.
        blocks = self.get_all_free_blocks()
        for block in blocks:
            block.eviction_priority = -1
        self._hashless = FreeKVCacheBlockQueue([])
        self._buckets = {}
        self._num_free = 0
        self.insert_n(blocks)

    def __len__(self) -> int:
        return self._num_free


# S3-FIFO queue labels stored in ``KVCacheBlock.s3_queue``. 0 = none: the block
# is hashless (worthless) or not resident. 1/2 = the block's content currently
# lives in the small / main FIFO. The label persists across in-use<->free
# transitions and, for a free block, also selects which free list holds it.
_S3_NONE = 0
_S3_SMALL = 1
_S3_MAIN = 2

DEFAULT_S3_GHOST_FACTOR = 1.0
DEFAULT_S3_PROMOTE_THRESHOLD = 1
DEFAULT_S3_FREQ_CAP = 3


class S3FIFOGPUCachePolicy(GPUCachePolicy):
    """S3-FIFO eviction: three FIFO queues with quick demotion.

    S3-FIFO (Yang et al., SOSP'23) admits new content to a small probationary
    FIFO; content re-accessed there is promoted to a main FIFO, otherwise it is
    quickly demoted and only its fingerprint is remembered in a ghost queue. If
    ghosted content is seen again it is admitted straight to main. This filters
    out one-hit-wonders cheaply while protecting proven-hot content.

    Adapting it to the GPU block pool: this policy, like the others here, only
    ever holds *free* blocks (``ref_cnt == 0``) -- an in-use block is off-limits
    (its KV is being read) so it is not in any queue -- and eviction is
    demand-driven (``evict_n`` when the scheduler needs blocks). The three FIFOs
    therefore contain only free blocks. Rather than the classic capacity split
    (evict from small once it exceeds a fixed fraction of the whole cache),
    eviction simply drains the small free list before the main free list: the
    free small blocks are the probationary candidates, so reusing them first
    needs no size target and no residency counters. Promotion and the ghost keep
    the S3-FIFO behavior; only the size-ratio bookkeeping is dropped.

    Layout (all hot-path operations O(1); ``evict_n`` is O(n) amortized as the
    frequency cap bounds second chances and promotions are bounded by the small
    free queue):

    - ``_hashless`` / ``_small`` / ``_main``: ``FreeKVCacheBlockQueue`` s of free
      blocks. Each is the intrusive O(1) doubly-linked list used elsewhere;
      newly freed blocks append at the tail, eviction pops the head (oldest).
      Worthless (hashless) blocks are drained first, then small, then main.
      Because a block leaves the free set while in use and re-appends at the
      tail when freed, order within a queue is "least recently freed", not
      classic insertion order. This is benign: a one-hit-wonder is never hit, so
      it is never reordered and still marches to the head for quick demotion; a
      reaccessed block is relabeled to main on its qualifying hit and files into
      the main queue when next freed.
    - ``_ghost``: an ``OrderedDict`` used as a FIFO set of content hashes demoted
      from the small queue (fingerprints only, no blocks). Bounded by
      ``round(ghost_factor * capacity)`` entries; the oldest drops first.

    Per block: ``KVCacheBlock.s3_queue`` (the small/main/none label, which also
    picks the free list a free block sits in) and ``KVCacheBlock.s3_freq`` (the
    frequency/hotness counter, capped at ``freq_cap``). Both reset when the block
    is reused for new content.

    Replicas (one content hash, multiple distinct physical blocks) are handled
    naturally: frequency and the small/main label live on each block, so
    replicas are promoted and evicted independently. Only the ghost is
    content-keyed, so replicas share one fingerprint, which is harmless.

    Staleness resistance: main uses a bounded second chance -- a hit bumps
    ``s3_freq`` (capped at ``freq_cap``) and each eviction pass that skips a
    block decrements it, so a block that stops being hit is evicted within
    ``freq_cap`` passes. Frequency lives on the block and resets on reuse (no
    long-term frequency table), so content is never pinned by stale popularity.

    Caveats (mirroring the other policies):

    - ``touch`` bumps the block's frequency and, once it crosses
      ``promote_threshold``, relabels a small block to main. That is only a label
      flip (the block is out of the free queue during a hit), so on its next
      free it files into main. ``BlockPool`` removes a block from the candidate
      set before recording a hit.
    - The KV-connector force-eviction path (``BlockPool.evict_blocks``) is not
      routed here. A block whose content is invalidated while it is a free
      candidate keeps its label until it is reused; ``insert`` clears the stale
      label when such a block is later freed as hashless.
    """

    def __init__(
        self,
        ghost_factor: float = DEFAULT_S3_GHOST_FACTOR,
        promote_threshold: int = DEFAULT_S3_PROMOTE_THRESHOLD,
        freq_cap: int = DEFAULT_S3_FREQ_CAP,
    ) -> None:
        assert ghost_factor >= 0.0
        assert 1 <= promote_threshold <= freq_cap
        self._ghost_factor = ghost_factor
        self._promote_threshold = promote_threshold
        self._freq_cap = freq_cap
        # Ghost bound; filled in by set_pool_capacity. A disabled ghost (0) is a
        # safe default: the only pre-capacity call is the all-hashless startup
        # insert, which never touches the ghost.
        self._ghost_capacity = 0
        self._hashless = FreeKVCacheBlockQueue([])
        self._small = FreeKVCacheBlockQueue([])
        self._main = FreeKVCacheBlockQueue([])
        self._ghost: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        capacity = max(num_gpu_blocks, 1)
        self._ghost_capacity = max(round(self._ghost_factor * capacity), 0)

    def _queue_for(self, block: KVCacheBlock) -> FreeKVCacheBlockQueue:
        # Which free list a free block belongs in, from its persistent label.
        if block.s3_queue == _S3_SMALL:
            return self._small
        if block.s3_queue == _S3_MAIN:
            return self._main
        return self._hashless

    def _admit(self, block: KVCacheBlock) -> None:
        # Label a fresh block's queue. A ghost hit (content previously demoted
        # from small) enters main; otherwise it enters small.
        block.s3_freq = 0
        block_hash = block.block_hash
        if block_hash is not None and block_hash in self._ghost:
            del self._ghost[block_hash]
            block.s3_queue = _S3_MAIN
        else:
            block.s3_queue = _S3_SMALL

    def _ghost_add(self, block_hash: BlockHashWithGroupId) -> None:
        if self._ghost_capacity <= 0:
            return
        if block_hash in self._ghost:
            return  # FIFO: keep the original position
        self._ghost[block_hash] = None
        while len(self._ghost) > self._ghost_capacity:
            self._ghost.popitem(last=False)  # drop the oldest fingerprint

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.insert(block)

    def insert(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            # Worthless: evict before any cached block. Clear any stale label (a
            # connector force-evicted the content while it was resident) so it
            # routes to hashless and remove() later targets the right queue.
            block.s3_queue = _S3_NONE
            block.s3_freq = 0
            self._hashless.append(block)
        else:
            if block.s3_queue == _S3_NONE:
                # Defensive: cached content that never went through on_cache
                # (the normal admission point while the block is in use).
                self._admit(block)
            self._queue_for(block).append(block)  # tail = most recently freed

    def remove(self, block: KVCacheBlock) -> None:
        # No-op if the block is not a free candidate (mirror LRU: an unlinked
        # block has no neighbors). The label/freq are kept: the block is being
        # reused by a hit, stays resident, so its residency count is unchanged.
        if block.prev_free_block is None or block.next_free_block is None:
            return
        self._queue_for(block).remove(block)  # O(1) via the block's own links

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        if block.block_hash is None:
            return
        block.s3_freq = min(block.s3_freq + 1, self._freq_cap)
        # Eager promotion: reaccessed small content graduates to main the moment
        # it crosses the threshold. The block is out of the free queue here
        # (BlockPool.touch removes it if it was free; an in-use block is in no
        # queue), so this is just a label flip, not a linked-list move.
        if block.s3_queue == _S3_SMALL and block.s3_freq >= self._promote_threshold:
            block.s3_queue = _S3_MAIN
            block.s3_freq = 0  # fresh start on main's second-chance budget

    def on_cache(self, block: KVCacheBlock, request_priority: int) -> None:
        # Admission: fresh content cached into an in-use block (never goes
        # through touch). Admit once per residency; a re-affirming call on an
        # already-labeled block is a no-op.
        if block.s3_queue != _S3_NONE or block.block_hash is None:
            return
        self._admit(block)

    def _evict_one_from_small(self) -> KVCacheBlock | None:
        # A free small block is always un-promoted (touch relabels it to main on
        # crossing the threshold), so small-eviction always demotes: return the
        # oldest block and remember its fingerprint in the ghost. None if empty.
        if self._small.num_free_blocks == 0:
            return None
        block = self._small.popleft()
        block_hash = block.block_hash
        if block_hash is not None:
            self._ghost_add(block_hash)
        block.s3_queue = _S3_NONE
        block.s3_freq = 0
        return block

    def _evict_one_from_main(self) -> KVCacheBlock | None:
        # Pop oldest main blocks: give a second chance to warm ones (decrement
        # and recycle) and return the first cold one. None only if main is empty.
        while self._main.num_free_blocks > 0:
            block = self._main.popleft()
            if block.s3_freq >= 1:
                block.s3_freq -= 1
                self._main.append(block)  # second chance
                continue
            block.s3_queue = _S3_NONE
            block.s3_freq = 0
            return block
        return None

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        if n <= 0:
            return []
        available = len(self)
        if available < n:
            raise ValueError(f"Cannot evict {n} blocks; only {available} are free")
        victims: list[KVCacheBlock] = []
        # Worthless (hashless) blocks first: no reuse value.
        take = min(self._hashless.num_free_blocks, n)
        victims.extend(self._hashless.popleft_n(take))
        # Then small (probation) before main (protected); each returns None only
        # when its own free queue is empty, so the other picks up the slack.
        while len(victims) < n:
            victim = self._evict_one_from_small() or self._evict_one_from_main()
            assert victim is not None, "no free block to evict despite len >= n"
            victims.append(victim)
        return victims

    def _ordered_queues(self) -> list[tuple[int, FreeKVCacheBlockQueue]]:
        # Approximate eviction order: hashless, then small, then main. evict_n is
        # the authoritative order (promotions/second chances make it dynamic);
        # this is the static best-effort order for sweeps and introspection.
        return [
            (_S3_NONE, self._hashless),
            (_S3_SMALL, self._small),
            (_S3_MAIN, self._main),
        ]

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        result: list[KVCacheBlock] = []
        for _, queue in self._ordered_queues():
            result.extend(queue.get_all_free_blocks())
        return result

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        # Resume within start_after's queue if it is still a free candidate;
        # otherwise sweep every queue from the best candidate onward.
        resume_label: int | None = None
        resume_node: KVCacheBlock | None = None
        if (
            start_after is not None
            and start_after.prev_free_block is not None
            and start_after.next_free_block is not None
        ):
            resume_label = start_after.s3_queue
            resume_node = start_after.next_free_block
        started = resume_label is None
        for label, queue in self._ordered_queues():
            if not started:
                if label != resume_label:
                    continue
                started = True
                node = resume_node
            else:
                node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            while node is not None and node is not tail:
                nxt = node.next_free_block  # capture before caller mutates node
                yield node
                node = nxt

    def reset(self) -> None:
        # Called after BlockPool clears every block hash, so all candidates are
        # now hashless. Drop all S3-FIFO state and re-home blocks into hashless.
        blocks = self.get_all_free_blocks()
        for block in blocks:
            block.s3_queue = _S3_NONE
            block.s3_freq = 0
        self._hashless = FreeKVCacheBlockQueue([])
        self._small = FreeKVCacheBlockQueue([])
        self._main = FreeKVCacheBlockQueue([])
        self._ghost.clear()
        self.insert_n(blocks)

    def __len__(self) -> int:
        return (
            self._hashless.num_free_blocks
            + self._small.num_free_blocks
            + self._main.num_free_blocks
        )


class TwoQueueGPUCachePolicy(GPUCachePolicy):
    """
    a block sits in cold on its first free, and moves to hot on its second entry into the free queue (or after any touch).
    """
    def __init__(self, ghost_factor = DEFAULT_GHOST_CAPACITY_FACTOR):
        # records whether a block_id is more frequently accessed (is a hot block) or not.
        # If the block_id was never in the free queue or was evicted it's block_id will not be there in _is_hot_candidate
        # At any point of time, if a block needs to be inserted into the cold queue, it's
        # id will not be there in is_hot_candidate
        self._is_hot_candidate: OrderedDict[int, bool] = OrderedDict() # block_id level boolean map for a hot candidate

        self._cold_queue = FreeKVCacheBlockQueue([])
        self._hot_queue = FreeKVCacheBlockQueue([])
        self._ghost_queue = OrderedDict[BlockHashWithGroupId, bool]([])
        # The ghost list remembers evicted content so that when the same content returns it can
        # skip cold. After eviction the physical block is repurposed (_maybe_evict_cached_block → reset_hash()), so block_id no longer
        # identifies that content — the only stable identity for "this content was recently evicted" is the hash. So ghost must be
        # hash-keyed.
        self._ghost_factor = ghost_factor if ghost_factor else DEFAULT_GHOST_CAPACITY_FACTOR
        self._ghost_capacity = 0

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        capacity = max(num_gpu_blocks, 1)
        self._ghost_capacity = max(round(self._ghost_factor * capacity), 0)

    def __is_block_in_use(self, block: KVCacheBlock) -> bool:
        """
        Determines whether or not a block is in use.
        The remove method in FreeKVCacheBlockQueue sets block.prev_free_block and block.next_free_block to None
        once the block is in use in the original vLLM implmentation. This function checks the same.
        """
        return block.prev_free_block is None and block.next_free_block is None


    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        """
        ASSUMPTION: for all blocks passed in the list parameter, remove is called before.
        That is it is not in any of the hot_queue or cold_queue
        """
        for block in blocks:
            block_hash = block.block_hash
            block_id = block.block_id
            if block_hash is None:
                # No hash => incomplete block => need to be freed first
                self.insert(block)
            else:
                if block_hash in self._ghost_queue:
                    # was evicted from the cold queue before
                    # move to the hot queue
                    del self._ghost_queue[block_hash] # remove from ghost queue, once the block is pushed to hot queue
                    self._is_hot_candidate[block_id] = True
                # elif block_hash not in self._is_hot_candidate:
                elif block_id not in self._is_hot_candidate:
                    # was never freed before. first time block.
                    # move to cold queue
                    self._is_hot_candidate[block_id] = False
                else:
                    # was freed before but not evicted
                    # move to hot queue
                    self._is_hot_candidate[block_id] = True

                self.insert(block) # will insert in hot/cold queue based on self._is_hot_candidate


    def insert(self, block: KVCacheBlock) -> None:
        block_hash = block.block_hash
        if block_hash is None:
            # No hash => incomplete block => need to be freed first
            self._cold_queue.prepend_n([block])
            return

        block_id = block.block_id
        if self._is_hot_candidate[block_id]:
            # insert into hot queue
            self._hot_queue.append(block)
            return
        # insert into cold queue
        self._cold_queue.append(block)


    def remove(self, block: KVCacheBlock) -> None:
        """Drop a block from the candidate set (it is being reused).
        function called when the block's ref_cnt changes: block.ref_cnt = 0 -> 1

        Must be a no-op if the block is not currently a candidate.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # ASSUMPTION: remove() never called on uncached blocks (block_hash = None)
            return

        block_id = block.block_id
        if block_id not in self._is_hot_candidate:
            print("WARNING: Request received for removing a block that was never in the free queue.")
            return
        if self.__is_block_in_use(block):
            # block in use
            return
        else:
            # block not in use
            if self._is_hot_candidate[block_id]:
                # resident of the hot queue, hence remove from there
                self._hot_queue.remove(block)
            else:
                # resident of cold queue, hence remove from there
                # ASSUMPTION: touch is called after remove. Hence when block is in
                # cold queue the access_cnt will be zero.
                self._cold_queue.remove(block)


    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        """Record a cache hit on a block (accounting only, no reordering)."""
        block_id = block.block_id
        self._is_hot_candidate[block_id] = True



    def evict_n(self, n: int) -> list[KVCacheBlock]:
        """Remove and return the ``n`` best eviction candidates."""
        cold_free_blocks = self._cold_queue.get_all_free_blocks()
        victims = []
        cold_victims = []
        n_cold_victims = n

        if n <= len(cold_free_blocks):
            # Only eviction from cold queue is required.
            cold_victims = cold_free_blocks[:n_cold_victims]
            victims = cold_free_blocks[:n_cold_victims]
            self._cold_queue.popleft_n(n_cold_victims)
        else:
            # eviction from both queues required
            # Collect len(cold_free_blocks) from the cold queue first to prioiritise hot blocks
            n_cold_victims = len(cold_free_blocks)
            cold_victims = cold_free_blocks
            victims = cold_free_blocks
            self._cold_queue.popleft_n(n_cold_victims)

            # To remove the remaining blocks from the hot queue
            hot_free_blocks = self._hot_queue.get_all_free_blocks()
            if len(hot_free_blocks) >= n - n_cold_victims:
                # hot queue has sufficient blocks to evict
                # evicted like this as hot blocks will be demoted to the cold queue
                # and will need to be evicted as it is to fit the capacity.
                hot_victims = hot_free_blocks[:n - n_cold_victims]
                victims = cold_victims + hot_victims
                self._hot_queue.popleft_n(n - n_cold_victims)
            else:
                # hot queue doesn't have enough blocks to evict
                raise ValueError(f"Tried to evict {n} free blocks. Evicted {n_cold_victims} blocks from the cold queue. Cannot evict any more blocks, as hot queue (num_free_blocks: {len(hot_free_blocks)}) doesn't have enough blocks.")

        # resetting block_level flags
        for cold_victim in cold_victims:
            cold_victim_hash = cold_victim.block_hash
            if cold_victim_hash is None:
                continue
            if len(self._ghost_queue) >= self._ghost_capacity:
                self._ghost_queue.popitem(last=False)   # drop oldest ghost
            # add to ghost queue
            self._ghost_queue[cold_victim_hash] = True

        for victim in victims:
            victim_block_id = victim.block_id
            # Guard against hashless cold queue filling with empty hot queue case.
            if victim_block_id in self._is_hot_candidate:
                del self._is_hot_candidate[victim_block_id] # remove elements from _is_hot_candidate
        return victims


    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Return all candidates in eviction order (best candidate first)."""
        cold_free_blocks = self._cold_queue.get_all_free_blocks()
        hot_free_blocks = self._hot_queue.get_all_free_blocks()
        return cold_free_blocks + hot_free_blocks


    def reset(self) -> None:
        """Forget any accumulated cache statistics (prefix-cache reset)."""
        self._is_hot_candidate = OrderedDict.fromkeys(self._is_hot_candidate, False) # All blocks are now cold.
        self._ghost_queue.clear()

        # move everything to cold queue
        hot_blocks = self._hot_queue.popleft_n(self.get_num_free_hot_blocks())
        for hot_block in hot_blocks:
            self._cold_queue.append(hot_block)


    def as_free_hot_block_queue(self) -> FreeKVCacheBlockQueue:
        """Return the underlying hot queue"""
        return self._hot_queue


    def as_free_cold_block_queue(self) -> FreeKVCacheBlockQueue:
        """Return the underlying cold queue"""
        return self._cold_queue


    def get_num_free_hot_blocks(self) -> int:
        """Number of eviction candidates currently held in hot queue."""
        return self._hot_queue.num_free_blocks


    def get_num_free_cold_blocks(self) -> int:
        """Number of eviction candidates currently held in cold queue."""
        return self._cold_queue.num_free_blocks


    def __len__(self) -> int:
        """Number of eviction candidates currently held in hot and cold queue."""
        return self._hot_queue.num_free_blocks + self._cold_queue.num_free_blocks


    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        """Yield free blocks in eviction order (best candidate first).

        ``start_after`` requests a resumable cursor: when the block is still a
        free candidate, iteration begins just after it, letting a caller sweep
        the free set incrementally across calls without rescanning from the
        front. When ``start_after`` is ``None`` or no longer free, iteration
        begins at the best candidate.

        The base implementation is correct but O(n) in the number of free
        blocks; policies with a traversable structure (LRU, LFU) override it to
        honor the cursor in O(1) per step.
        """
        cold_free_blocks = self._cold_queue.get_all_free_blocks()
        hot_free_blocks = self._hot_queue.get_all_free_blocks()
        free_blocks = cold_free_blocks + hot_free_blocks
        start = 0
        if start_after is not None:
            for i, block in enumerate(free_blocks):
                if block is start_after:
                    start = i + 1
                    break
        yield from free_blocks[start:]


class ARCGPUCachePolicy(GPUCachePolicy):
    """ARC (Adaptive Replacement Cache) eviction for the GPU KV cache.

    ARC (Megiddo & Modha, FAST '03) self-tunes the balance between *recency* and
    *frequency*. Content seen once lives in T1 (recency); content seen again lives
    in T2 (frequency). Ghost lists B1/B2 remember the hashes of content recently
    evicted from T1/T2 (no data). The adaptive target ``p`` for the T1 partition is
    grown on a B1 ghost hit (recency mattered) and shrunk on a B2 ghost hit
    (frequency mattered). See ``ARC_EVICTION_PLAN.md`` for the full mapping.

    vLLM adaptation (see plan §2, §4):

    - The policy holds only *free* blocks, so T1/T2 are tracked at two levels: the
      **logical** sizes ``_n_t1`` / ``_n_t2`` count all content-holding blocks
      (in-use **+** free) and drive the eviction decision, while three
      ``FreeKVCacheBlockQueue`` s (``_hashless`` / ``_t1_free`` / ``_t2_free``)
      hold only the free subset for O(1) physical eviction. Using logical sizes is
      essential: comparing a floating free-only ``|T1|`` against a capacity-scaled
      ``p`` would be meaningless.
    - The ARC reference events map to hooks: a cache hit is ``touch`` (Case I,
      promote to T2); a fresh cache is ``on_cache`` (Case IV → T1, or a B1/B2 ghost
      hit → adapt ``p`` and enter T2). ``p`` adaptation happens here, at admission.
    - REPLACE (the "check sizes, evict from T1 or T2" subroutine) runs at
      ``evict_n`` -- vLLM's only physical eviction point -- because eviction is
      demand-driven (a block leaves the cache only when the allocator needs it).
      A block's list label persists on ``KVCacheBlock.arc_list`` across
      free<->in-use transitions.

    Per-block state lives in ``KVCacheBlock.arc_list`` (0 = none/hashless, 1 = T1,
    2 = T2); it selects the free sub-queue while free and keeps the logical counts
    consistent while in use.
    """

    def __init__(self) -> None:
        # Cache size c; filled in by set_pool_capacity (constant = num_gpu_blocks).
        self._c = 1
        self._p = 0.0  # adaptive target size for T1, in [0, c]
        # Logical partition sizes: count in-use + free blocks per list.
        self._n_t1 = 0
        self._n_t2 = 0
        # Free sub-queues (evictable candidates only). Hashless drained first.
        self._hashless = FreeKVCacheBlockQueue([])
        self._t1_free = FreeKVCacheBlockQueue([])
        self._t2_free = FreeKVCacheBlockQueue([])
        # Ghost lists: fingerprints (hashes) of content recently evicted from
        # T1/T2. OrderedDict = FIFO set; oldest trimmed first.
        self._b1: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self._b2: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        self._c = max(num_gpu_blocks, 1)

    def _set_label(self, block: KVCacheBlock, new_label: int) -> None:
        # Move a block between logical lists, keeping _n_t1 / _n_t2 exact. This is
        # the ONLY place the counters change (admit, promote, evict, reconcile).
        old = block.arc_list
        if old == new_label:
            return
        if old == 1:
            self._n_t1 -= 1
        elif old == 2:
            self._n_t2 -= 1
        if new_label == 1:
            self._n_t1 += 1
        elif new_label == 2:
            self._n_t2 += 1
        block.arc_list = new_label

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            self.insert(block)

    def insert(self, block: KVCacheBlock) -> None:
        # A freed block re-enters the free sub-queue named by its label. No ghost
        # logic here (adaptation happens at on_cache); no counter change for a
        # cached block (already counted at admission).
        if block.block_hash is None:
            # Worthless / content gone (e.g. connector force-eviction). Reconcile
            # any stale label, then evict-first (front of the hashless queue).
            self._set_label(block, 0)
            self._hashless.prepend_n([block])
            return
        if block.arc_list == 2:
            self._t2_free.append(block)  # tail = most recently freed
        else:
            # arc_list == 1 (T1) or, defensively, 0 (never admitted) -> count it.
            if block.arc_list == 0:
                self._set_label(block, 1)
            self._t1_free.append(block)

    def remove(self, block: KVCacheBlock) -> None:
        # No-op if the block is not a free candidate (mirror the sibling policies:
        # an unlinked block has both neighbor links cleared). Keep the label and
        # the logical counts: the block goes in use but stays logically in its list.
        if block.prev_free_block is None or block.next_free_block is None:
            return
        label = block.arc_list
        if label == 2:
            self._t2_free.remove(block)
        elif label == 1:
            self._t1_free.remove(block)
        else:
            self._hashless.remove(block)

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        # Case I: a cache hit promotes content to the frequency list T2. The block
        # is out of the free queue here (BlockPool.touch removes it first), so this
        # is only a label/count move; it files into T2 when next freed.
        if block.block_hash is None:
            return
        self._set_label(block, 2)

    def on_cache(self, block: KVCacheBlock, request_priority: int) -> None:
        # Admission: fresh content cached into an in-use block. This is the ARC
        # reference where ghost hits are detected and ``p`` adapts.
        hash_key = block.block_hash
        if hash_key is None:
            return
        if hash_key in self._b1:
            # Case II (B1 ghost hit): recency mattered -> grow p, enter T2.
            delta = max(1.0, len(self._b2) / len(self._b1))
            self._p = min(self._p + delta, float(self._c))
            del self._b1[hash_key]
            self._set_label(block, 2)
        elif hash_key in self._b2:
            # Case III (B2 ghost hit): frequency mattered -> shrink p, enter T2.
            delta = max(1.0, len(self._b1) / len(self._b2))
            self._p = max(self._p - delta, 0.0)
            del self._b2[hash_key]
            self._set_label(block, 2)
        else:
            # Case IV (true miss): new content enters the recency list T1.
            self._set_label(block, 1)

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        # REPLACE, realised at vLLM's demand-driven eviction point.
        if n <= 0:
            return []
        available = len(self)
        if available < n:
            raise ValueError(f"Cannot evict {n} blocks; only {available} are free")
        victims: list[KVCacheBlock] = []
        # Worthless (hashless) blocks first: no reuse value, no ghost.
        take = min(self._hashless.num_free_blocks, n)
        victims.extend(self._hashless.popleft_n(take))
        # Then the ARC replacement rule on the logical sizes. _set_label keeps
        # _n_t1 / _n_t2 exact as we go, so the next decision sees updated sizes.
        while len(victims) < n:
            prefer_t1 = self._n_t1 > self._p
            if prefer_t1 and self._t1_free.num_free_blocks > 0:
                block, ghost = self._t1_free.popleft(), self._b1
            elif not prefer_t1 and self._t2_free.num_free_blocks > 0:
                block, ghost = self._t2_free.popleft(), self._b2
            elif self._t1_free.num_free_blocks > 0:
                # Preferred list has no free block (all pinned) -> fall back, and
                # attribute the eviction to the list actually touched.
                block, ghost = self._t1_free.popleft(), self._b1
            else:
                block, ghost = self._t2_free.popleft(), self._b2
            hash_key = block.block_hash
            if hash_key is not None:
                ghost[hash_key] = None  # record fingerprint before hash is reset
            self._set_label(block, 0)
            victims.append(block)
        self._trim_ghosts()
        return victims

    def _trim_ghosts(self) -> None:
        # Bound each ghost list to c entries (drop oldest first).
        while len(self._b1) > self._c:
            self._b1.popitem(last=False)
        while len(self._b2) > self._c:
            self._b2.popitem(last=False)

    def _ordered_queues(self) -> list[FreeKVCacheBlockQueue]:
        # Approximate eviction order: hashless, then T1 (recency), then T2. evict_n
        # is authoritative (the T1/T2 split is dynamic); this is best-effort.
        return [self._hashless, self._t1_free, self._t2_free]

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        result: list[KVCacheBlock] = []
        for queue in self._ordered_queues():
            result.extend(queue.get_all_free_blocks())
        return result

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        # Resume within start_after's queue if it is still a free candidate;
        # otherwise sweep every queue from the best candidate onward.
        resume_queue: FreeKVCacheBlockQueue | None = None
        resume_node: KVCacheBlock | None = None
        if (
            start_after is not None
            and start_after.prev_free_block is not None
            and start_after.next_free_block is not None
        ):
            label = start_after.arc_list
            resume_queue = (
                self._t2_free
                if label == 2
                else self._t1_free
                if label == 1
                else self._hashless
            )
            resume_node = start_after.next_free_block
        started = resume_queue is None
        for queue in self._ordered_queues():
            if not started:
                if queue is not resume_queue:
                    continue
                started = True
                node = resume_node
            else:
                node = queue.fake_free_list_head.next_free_block
            tail = queue.fake_free_list_tail
            while node is not None and node is not tail:
                nxt = node.next_free_block  # capture before caller mutates node
                yield node
                node = nxt

    def reset(self) -> None:
        # BlockPool cleared every block hash, so all free candidates are now
        # hashless. Clear their labels (keeping the counters exact for any pinned
        # blocks, which reconcile when freed), drop ghosts, and re-home them.
        blocks = self.get_all_free_blocks()
        for block in blocks:
            self._set_label(block, 0)
        self._p = 0.0
        self._hashless = FreeKVCacheBlockQueue([])
        self._t1_free = FreeKVCacheBlockQueue([])
        self._t2_free = FreeKVCacheBlockQueue([])
        self._b1.clear()
        self._b2.clear()
        self.insert_n(blocks)

    def __len__(self) -> int:
        # Free eviction candidates (not the logical n_t1 + n_t2, which include
        # pinned blocks); this is the count the block pool expects.
        return (
            self._hashless.num_free_blocks
            + self._t1_free.num_free_blocks
            + self._t2_free.num_free_blocks
        )

    # --- diagnostics / testing helpers ---

    @property
    def p(self) -> float:
        return self._p

    @property
    def n_t1(self) -> int:
        return self._n_t1

    @property
    def n_t2(self) -> int:
        return self._n_t2

    @property
    def b1_size(self) -> int:
        return len(self._b1)

    @property
    def b2_size(self) -> int:
        return len(self._b2)


# Defaults for the adaptive main-gap predictor (see AdaptiveMainGapGPUCachePolicy).
DEFAULT_ADAPTIVE_MARGIN = 0.0  # switch on any positive edge (kvsim default)
DEFAULT_ADAPTIVE_INTERVAL_MIN = 15.0  # wall-clock minutes between decisions
DEFAULT_ADAPTIVE_MIN_REFERENCES = 100  # min refs in a window to weigh a switch
DEFAULT_ADAPTIVE_HISTORY_FACTOR = 2.0  # tracked-hash budget as a multiple of C
# Optional absolute ceiling on the tracked-hash budget H (default: no cap, so
# H = round(history_factor * capacity)). Only useful for very large pools, where
# an uncapped H makes the O(H) compaction burst and Fenwick depth grow with the
# pool. Must stay >= the effective capacity C or LRU-hit detection is truncated.
DEFAULT_ADAPTIVE_HISTORY_MAX: int | None = None
DEFAULT_ADAPTIVE_CONSECUTIVE = 1  # windows a challenger must win (1 = none)


class _MainGapPredictor:
    """Analytical LRU-vs-S3-FIFO hit-rate estimator (kvsim 'flat' reach model).

    Fed one content hash per block *reference* (a hit or a miss-fill). For each
    reuse it estimates, in O(log H), whether LRU and S3-FIFO would have held the
    content, tallying per-window ``_win_lru`` / ``_win_s3`` over ``_win_reuse``
    reuse events. ``decide()`` returns the two estimated hit rates.

    Reuse distance uses two Fenwick trees over dense access positions:

    - ``_bit``: +1 at the latest position of EVERY tracked hash.
    - ``_bit_hot``: +1 at the latest position of HOT hashes (refcnt >= 2).

    ``dist`` = distinct hashes referenced since a hash's previous access;
    ``hot_in`` = distinct HOT hashes since then. ``dist - hot_in`` is the cold
    (one-shot) mass S3-FIFO's small queue filters.

    Memory is bounded: at most ``history_capacity`` tracked hashes (oldest
    evicted first) and positions compacted into ``[1..pos_capacity]``. This is
    the only algorithmic addition over the kvsim source; content whose reuse
    distance already exceeds C is a miss under both policies, so dropping it
    loses no signal.
    """

    def __init__(
        self,
        history_factor: float = DEFAULT_ADAPTIVE_HISTORY_FACTOR,
        history_max: int | None = DEFAULT_ADAPTIVE_HISTORY_MAX,
    ) -> None:
        assert history_factor >= 1.0
        assert history_max is None or history_max >= 1
        self._history_factor = history_factor
        self._history_max = history_max
        # Sized in set_pool_capacity(); safe bootstrap values until then.
        self._history_capacity = 1
        self._pos_capacity = 2
        self._C = 1.0
        self._sigma = 0.5
        self._mu = 0.5
        self._reset_state()

    def _reset_state(self) -> None:
        # recency of last access: hash -> dense position, ordered oldest-first.
        self._pos: OrderedDict[BlockHashWithGroupId, int] = OrderedDict()
        self._refcnt: dict[BlockHashWithGroupId, int] = {}
        self._bit = [0] * (self._pos_capacity + 1)
        self._bit_hot = [0] * (self._pos_capacity + 1)
        self._t = 0
        self._hot_total = 0
        self._reset_window()
        # per-window running capacity samples
        self._cap_sum = 0.0
        self._cap_n = 0
        # C is seeded from the first live sample so the very first window already
        # classifies at effective capacity (no full-pool warmup window).
        self._c_seeded = False

    def _reset_window(self) -> None:
        self._win_lru = 0
        self._win_s3 = 0
        self._win_reuse = 0

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        cap = max(int(num_gpu_blocks), 1)
        self._set_capacity(cap)
        h = round(self._history_factor * cap)
        if self._history_max is not None:
            h = min(h, self._history_max)
        self._history_capacity = max(h, 1)
        self._pos_capacity = 2 * self._history_capacity
        self._reset_state()

    def _set_capacity(self, capacity: float) -> None:
        self._C = max(float(capacity), 1.0)
        self._sigma = 0.5 * self._C
        self._mu = 0.5 * self._C

    @staticmethod
    def _bit_update(bit: list[int], i: int, d: int) -> None:
        n = len(bit)
        while i < n:
            bit[i] += d
            i += i & (-i)

    @staticmethod
    def _bit_query(bit: list[int], i: int) -> int:
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    def _evict_oldest(self) -> None:
        # Drop the least-recently-referenced tracked hash and its Fenwick marks.
        old_h, old_pos = self._pos.popitem(last=False)
        self._bit_update(self._bit, old_pos, -1)
        if self._refcnt.get(old_h, 0) >= 2:
            self._bit_update(self._bit_hot, old_pos, -1)
            self._hot_total -= 1
        self._refcnt.pop(old_h, None)

    def _compact(self) -> None:
        # Re-densify live positions into [1..len]; rebuild both trees; reset _t.
        ordered = sorted(self._pos, key=self._pos.__getitem__)
        self._bit = [0] * (self._pos_capacity + 1)
        self._bit_hot = [0] * (self._pos_capacity + 1)
        self._hot_total = 0
        new_pos: OrderedDict[BlockHashWithGroupId, int] = OrderedDict()
        for i, h in enumerate(ordered, start=1):
            new_pos[h] = i
            self._bit_update(self._bit, i, 1)
            if self._refcnt.get(h, 0) >= 2:
                self._bit_update(self._bit_hot, i, 1)
                self._hot_total += 1
        self._pos = new_pos
        self._t = len(ordered)

    def sample_capacity(self, free_blocks: int) -> None:
        self._cap_sum += float(free_blocks)
        self._cap_n += 1
        if not self._c_seeded:
            self._c_seeded = True
            self._set_capacity(free_blocks)

    def record_ref(self, h: BlockHashWithGroupId) -> None:
        prev = self._pos.get(h)
        rc = self._refcnt.get(h, 0)
        was_hot = rc >= 2
        if prev is not None:
            dist = len(self._pos) - self._bit_query(self._bit, prev)
            hot_in = self._hot_total - self._bit_query(self._bit_hot, prev)
            self._bit_update(self._bit, prev, -1)  # drop h's old recency mark
            if was_hot:
                self._bit_update(self._bit_hot, prev, -1)
                self._hot_total -= 1
            self._classify(dist, hot_in, was_hot)
        elif len(self._pos) >= self._history_capacity:
            self._evict_oldest()  # bound tracked-hash count
        # bump refcnt (promotes to hot on first reuse), advance the recency
        # clock, and stamp h's new position BEFORE (re)marking, so a compaction
        # rebuild counts h exactly once (kvsim's set-position-then-grow order).
        self._refcnt[h] = rc + 1
        now_hot = (rc + 1) >= 2
        self._t += 1
        self._pos[h] = self._t
        self._pos.move_to_end(h)  # keep oldest-first order for eviction
        if self._t > self._pos_capacity:
            self._compact()  # rebuilds all marks from _pos (incl. h)
        else:
            self._bit_update(self._bit, self._t, 1)
            if now_hot:
                self._bit_update(self._bit_hot, self._t, 1)
                self._hot_total += 1

    def _classify(self, dist: int, hot_in: int, was_hot: bool) -> None:
        # One reuse: score it for LRU and the observed-reuse S3 model (flat).
        self._win_reuse += 1
        if dist <= self._C:  # LRU holds iff < C distinct blocks since
            self._win_lru += 1
        if was_hot:  # HOT (main, depth mu): only hot blocks compete
            if hot_in <= self._mu:
                self._win_s3 += 1
        else:  # COLD (small, depth sigma): only cold blocks compete
            if (dist - hot_in) <= self._sigma:
                self._win_s3 += 1

    def finalize_capacity(self) -> None:
        # Effective capacity = avg free-block count this window (mirrors the
        # controller's avg_free_blocks). Keep the prior if no samples.
        if self._cap_n > 0:
            self._set_capacity(self._cap_sum / self._cap_n)
        self._cap_sum = 0.0
        self._cap_n = 0

    def decide(self) -> tuple[float, float]:
        # Reuse-conditional hit-rate estimates (hits per reuse reference).
        if self._win_reuse == 0:
            return 0.0, 0.0
        return self._win_lru / self._win_reuse, self._win_s3 / self._win_reuse

    def window_reuse(self) -> int:
        """Number of reuse references classified in the current window."""
        return self._win_reuse

    def reset_window(self) -> None:
        self._reset_window()

    def reset(self) -> None:
        self._reset_state()


class DynamicGPUCachePolicy(GPUCachePolicy):
    """Runtime-switchable eviction: LRU <-> S3-FIFO with no state loss.

    Different workloads favor different policies, so this wrapper holds one LRU
    and one S3-FIFO sub-policy and delegates every interface call to whichever
    is currently *active*. An external controller flips the active policy at
    runtime via ``set_active_policy`` (wired through a control RPC).

    The two sub-policies cannot both be live at once: ``LRUGPUCachePolicy`` and
    ``S3FIFOGPUCachePolicy`` both thread free blocks through the *same* intrusive
    ``prev_free_block`` / ``next_free_block`` links on ``KVCacheBlock``, and a
    block can only sit in one linked list. So only the active policy owns the
    links; the inactive one is empty.

    To switch without losing history, the wrapper maintains two policy-agnostic
    per-block stats that are updated on *every* access regardless of which
    sub-policy is active:

    - ``KVCacheBlock.dyn_freq``: lifetime hit count (``+1`` per ``touch``, reset
      to 0 on fresh ``on_cache``).
    - ``KVCacheBlock.dyn_clock``: a monotonic free-time stamp (set on ``insert``,
      i.e. when the block is freed back into the pool). Not stamped on ``touch``
      because neither live policy reorders on access -- LRU ``touch`` is a no-op
      and S3 ``touch`` only bumps frequency -- so free-time is the faithful
      ordering key.

    A switch then drains the active policy's free blocks and rebuilds the target
    faithfully from those stats (see ``set_active_policy``). This rewires
    software bookkeeping only; no physical KV memory is moved and no in-use
    (pinned) block is touched.
    """

    def __init__(
        self,
        lru: LRUGPUCachePolicy,
        s3: S3FIFOGPUCachePolicy,
        start_policy: str = "lru",
    ) -> None:
        assert start_policy in ("lru", "s3_fifo")
        self._lru = lru
        self._s3 = s3
        self._policies: dict[str, GPUCachePolicy] = {"lru": lru, "s3_fifo": s3}
        self._active_name = start_policy
        self._active = self._policies[start_policy]
        self._clock = 0

    @property
    def active_policy(self) -> str:
        return self._active_name

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    # --- interface: delegate to the active sub-policy, updating universal stats

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        # Stamp free-time in list order (blocks[0] is the best eviction
        # candidate, so it gets the oldest stamp), then delegate.
        for block in blocks:
            block.dyn_clock = self._tick()
        self._active.insert_n(blocks)

    def remove(self, block: KVCacheBlock) -> None:
        self._active.remove(block)

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        if block.block_hash is not None:
            block.dyn_freq += 1
        self._active.touch(block, request_priority)

    def on_cache(self, block: KVCacheBlock, request_priority: int) -> None:
        # Fresh content in this physical block: its lifetime hit count restarts.
        block.dyn_freq = 0
        self._active.on_cache(block, request_priority)

    def evict_n(self, n: int) -> list[KVCacheBlock]:
        return self._active.evict_n(n)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        return self._active.get_all_free_blocks()

    def iter_eviction_order(
        self, start_after: KVCacheBlock | None = None
    ) -> Iterator[KVCacheBlock]:
        return self._active.iter_eviction_order(start_after)

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        # Both sub-policies must know the capacity (S3 sizes its ghost from it),
        # since either may become active later.
        self._lru.set_pool_capacity(num_gpu_blocks)
        self._s3.set_pool_capacity(num_gpu_blocks)

    def reset(self) -> None:
        # Prefix-cache reset clears every block hash, so popularity is moot.
        for block in self._active.get_all_free_blocks():
            block.dyn_freq = 0
        self._active.reset()

    def as_free_block_queue(self) -> FreeKVCacheBlockQueue:
        # Legacy call sites (CPU-offloading connector) require an LRU-style
        # queue; honor it when LRU is active, otherwise raise like S3 does.
        return self._active.as_free_block_queue()

    def __len__(self) -> int:
        return len(self._active)

    # --- the switch --------------------------------------------------------- #

    def set_active_policy(self, name: str) -> bool:
        """Switch the active eviction policy, preserving the free-block set.

        Rebuilds the target policy's ordering from the universal per-block
        stats, so nothing is lost. Fully synchronous and safe to call between
        scheduler steps (the engine core is single-threaded, so it cannot race
        an insert/evict). Returns ``True`` if the active policy changed.

        Raises:
            ValueError: if ``name`` is not ``"lru"`` or ``"s3_fifo"``.
        """
        name = name.lower()
        if name not in self._policies:
            raise ValueError(
                f"The 'dynamic' policy switches only between 'lru' and "
                f"'s3_fifo', got {name!r}."
            )
        if name == self._active_name:
            return False

        free_count = len(self._active)
        free = self._drain(self._active)
        assert len(free) == free_count, "drained block count out of sync"

        target = self._policies[name]
        if target is self._lru:
            self._rebuild_lru(free)
        else:
            self._rebuild_s3(free)

        self._active = target
        self._active_name = name
        assert len(self._active) == free_count, "rebuilt block count out of sync"
        return True

    def _drain(self, policy: GPUCachePolicy) -> list[KVCacheBlock]:
        # Pop every free block, clearing its intrusive links (via the policy's
        # own remove), so the drained policy ends empty (len == 0) and no block
        # stays threaded into it. get_all_free_blocks returns a snapshot list, so
        # removing while iterating it is safe.
        blocks = policy.get_all_free_blocks()
        for block in blocks:
            policy.remove(block)
        assert len(policy) == 0, "source policy not fully drained"
        return blocks

    def _rebuild_lru(self, free: list[KVCacheBlock]) -> None:
        # Recency order: oldest free-time first (front = evicted first). LRU's
        # own insert_n still routes hashless blocks to the front.
        self._lru.insert_n(sorted(free, key=lambda b: b.dyn_clock))

    def _rebuild_s3(self, free: list[KVCacheBlock]) -> None:
        # Partition-only reconstruction: use dyn_freq only to decide small vs.
        # main; do NOT seed s3_freq from it (live s3_freq is a decaying
        # second-chance budget, reset to 0 on promotion, not a lifetime count).
        # s3_freq = 0 makes each block a freshly-promoted one and lets S3 rebuild
        # its counters naturally. Ghost starts empty.
        self._s3._ghost.clear()
        threshold = self._s3._promote_threshold
        for block in free:
            if block.block_hash is None:
                block.s3_queue = _S3_NONE
            elif block.dyn_freq >= threshold:
                block.s3_queue = _S3_MAIN
            else:
                block.s3_queue = _S3_SMALL
            block.s3_freq = 0
        self._s3.insert_n(sorted(free, key=lambda b: b.dyn_clock))


class AdaptiveMainGapGPUCachePolicy(DynamicGPUCachePolicy):
    """Self-tuning LRU <-> S3-FIFO via an in-process analytical predictor.

    Ports kvsim's ``AdaptiveMainGapPolicy`` ('flat' reach model). Every block
    reference (a ``touch`` hit or an ``on_cache`` miss-fill) feeds a reuse-
    distance predictor. Every ``interval_min`` wall-clock minutes -- checked at
    the next block-free (``insert_n``), the safe switch boundary -- the policy
    compares the predicted LRU vs. S3-FIFO hit rates over the elapsed window
    and, if the challenger beats the incumbent by ``margin`` for ``consecutive``
    windows (and the window saw at least ``min_references`` references), switches
    -- reusing ``DynamicGPUCachePolicy``'s state-preserving swap. Unlike the
    ``"dynamic"`` policy, no controller thread, subprocess, or shadow trace is
    involved: the decision is computed inline on the engine-core thread, so a
    switch cannot race an insert/evict.
    """

    def __init__(
        self,
        lru: LRUGPUCachePolicy,
        s3: S3FIFOGPUCachePolicy,
        start_policy: str = "lru",
        *,
        margin: float = DEFAULT_ADAPTIVE_MARGIN,
        interval_min: float = DEFAULT_ADAPTIVE_INTERVAL_MIN,
        min_references: int = DEFAULT_ADAPTIVE_MIN_REFERENCES,
        history_factor: float = DEFAULT_ADAPTIVE_HISTORY_FACTOR,
        history_max: int | None = DEFAULT_ADAPTIVE_HISTORY_MAX,
        consecutive: int = DEFAULT_ADAPTIVE_CONSECUTIVE,
    ) -> None:
        super().__init__(lru, s3, start_policy)
        self._predictor = _MainGapPredictor(history_factor, history_max)
        self._margin = margin
        self._interval_s = max(1.0, interval_min * 60.0)
        self._min_references = min_references
        self._consecutive = consecutive
        self._wins = 0  # consecutive windows the (current) challenger has won
        self._last_decision_time = time.monotonic()

    def set_pool_capacity(self, num_gpu_blocks: int) -> None:
        super().set_pool_capacity(num_gpu_blocks)  # sizes both sub-policies
        self._predictor.set_pool_capacity(num_gpu_blocks)

    # --- hooks: feed the predictor, then delegate to the switch machinery ---

    def touch(self, block: KVCacheBlock, request_priority: int = 0) -> None:
        self._reference(block)  # a hit is a reference
        super().touch(block, request_priority)

    def on_cache(self, block: KVCacheBlock, request_priority: int) -> None:
        self._reference(block)  # a miss-fill is a reference
        super().on_cache(block, request_priority)

    def insert_n(self, blocks: list[KVCacheBlock]) -> None:
        # A clean boundary: the incoming blocks are not yet linked and the
        # existing free set is stable, so a switch (drain + rebuild) is safe.
        # Decide at most once per ``interval_min`` of wall-clock time; the check
        # is one monotonic() read, and insert_n fires on every block-free.
        now = time.monotonic()
        if now - self._last_decision_time >= self._interval_s:
            self._last_decision_time = now
            self._decide()
        super().insert_n(blocks)

    def reset(self) -> None:
        super().reset()
        self._predictor.reset()
        self._wins = 0
        self._last_decision_time = time.monotonic()

    # --- internals ---

    def _reference(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            return  # hashless/worthless: not a content reference
        # Effective-capacity sample: current free-candidate count.
        self._predictor.sample_capacity(len(self._active))
        self._predictor.record_ref(block.block_hash)

    def _decide(self) -> None:
        self._predictor.finalize_capacity()
        p_lru, p_s3 = self._predictor.decide()
        reuse = self._predictor.window_reuse()
        self._predictor.reset_window()  # recency/hotness PERSIST; tallies reset
        if reuse < self._min_references:
            self._wins = 0  # too little signal this window; do not switch
            return
        active = self._active_name
        if active == "lru":
            challenger, edge = "s3_fifo", p_s3 - p_lru
        else:
            challenger, edge = "lru", p_lru - p_s3
        if edge > self._margin:
            self._wins += 1
            if self._wins >= self._consecutive:
                self._wins = 0
                self.set_active_policy(challenger)  # inherited, state-preserving
        else:
            self._wins = 0

    def _rebuild_s3(self, free: list[KVCacheBlock]) -> None:
        # Non-destructive override of DynamicGPUCachePolicy._rebuild_s3. The base
        # clears S3's ghost and zeroes every block's s3_freq on a switch; because
        # S3's main eviction reclaims a freq-0 block immediately (no second
        # chance), that strands the whole rebuilt main queue -- including a reused
        # multi-turn prefix -- and this policy's frequent switches then collapse
        # the hit rate. Here the ghost is preserved and each main block's
        # second-chance budget is seeded from its lifetime hotness (bounded by
        # freq_cap), so a switch keeps S3's accumulated protection.
        threshold = self._s3._promote_threshold
        freq_cap = self._s3._freq_cap
        for block in free:
            if block.block_hash is None:
                block.s3_queue = _S3_NONE
                block.s3_freq = 0
            elif block.dyn_freq >= threshold:
                block.s3_queue = _S3_MAIN
                block.s3_freq = min(block.dyn_freq, freq_cap)
            else:
                block.s3_queue = _S3_SMALL
                block.s3_freq = 0
        self._s3.insert_n(sorted(free, key=lambda b: b.dyn_clock))


def _build_lru(extra_config: dict[str, Any]) -> GPUCachePolicy:
    if extra_config:
        raise ValueError(
            f"The 'lru' eviction policy takes no options, got "
            f"{sorted(extra_config)}."
        )
    return LRUGPUCachePolicy()


def _build_arc(extra_config: dict[str, Any]) -> GPUCachePolicy:
    # ARC is fully self-tuning (the adaptive target p needs no configuration).
    if extra_config:
        raise ValueError(
            f"The 'arc' eviction policy takes no options, got "
            f"{sorted(extra_config)}."
        )
    return ARCGPUCachePolicy()


def _build_lfu(extra_config: dict[str, Any]) -> GPUCachePolicy:
    known = {"max_freq", "ghost_capacity"}
    unknown = set(extra_config) - known
    if unknown:
        raise ValueError(
            f"Unknown options for the 'lfu' eviction policy: {sorted(unknown)}. "
            f"Supported options: {sorted(known)}."
        )
    max_freq = extra_config.get("max_freq", DEFAULT_LFU_MAX_FREQ)
    ghost_capacity = extra_config.get("ghost_capacity")
    _validate_positive_int(max_freq, "max_freq", allow_none=False)
    _validate_positive_int(ghost_capacity, "ghost_capacity", allow_none=True)
    return LFUGPUCachePolicy(max_freq=max_freq, ghost_capacity=ghost_capacity)


def _validate_positive_int(
    value: Any, name: str, *, allow_none: bool, policy: str = "lfu"
) -> None:
    if allow_none and value is None:
        return
    # bool is an int subclass; reject it so `max_freq: true` fails loudly.
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        suffix = " or omitted" if allow_none else ""
        raise ValueError(
            f"The {policy!r} option {name!r} must be an int >= 1{suffix}, "
            f"got {value!r}."
        )


def _validate_nonneg_number(value: Any, name: str, policy: str = "s3_fifo") -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(
            f"The {policy!r} option {name!r} must be a number >= 0, got {value!r}."
        )


def _build_priority(extra_config: dict[str, Any]) -> GPUCachePolicy:
    if extra_config:
        raise ValueError(
            f"The 'priority' eviction policy takes no options, got "
            f"{sorted(extra_config)}."
        )
    return PriorityGPUCachePolicy()


def _build_s3_fifo(extra_config: dict[str, Any]) -> GPUCachePolicy:
    known = {"ghost_factor", "promote_threshold", "freq_cap"}
    unknown = set(extra_config) - known
    if unknown:
        raise ValueError(
            f"Unknown options for the 's3_fifo' eviction policy: "
            f"{sorted(unknown)}. Supported options: {sorted(known)}."
        )
    ghost_factor = extra_config.get("ghost_factor", DEFAULT_S3_GHOST_FACTOR)
    promote_threshold = extra_config.get(
        "promote_threshold", DEFAULT_S3_PROMOTE_THRESHOLD
    )
    freq_cap = extra_config.get("freq_cap", DEFAULT_S3_FREQ_CAP)
    _validate_nonneg_number(ghost_factor, "ghost_factor")
    _validate_positive_int(
        promote_threshold, "promote_threshold", allow_none=False, policy="s3_fifo"
    )
    _validate_positive_int(freq_cap, "freq_cap", allow_none=False, policy="s3_fifo")
    if promote_threshold > freq_cap:
        raise ValueError(
            f"The 's3_fifo' option 'promote_threshold' ({promote_threshold}) "
            f"must be <= 'freq_cap' ({freq_cap})."
        )
    return S3FIFOGPUCachePolicy(
        ghost_factor=ghost_factor,
        promote_threshold=promote_threshold,
        freq_cap=freq_cap,
    )


def _build_twoqueue(extra_config: dict[str, Any]) -> GPUCachePolicy:
    """
    Instantiate the TwoQueue class
    """
    known = {"max_freq", "ghost_capacity"}
    unknown = set(extra_config) - known
    if unknown:
        raise ValueError(
            f"Unknown options for the '2q' eviction policy: {sorted(unknown)}. "
            f"Supported options: {sorted(known)}."
        )
    ghost_capacity = extra_config.get("ghost_capacity")
    _validate_positive_int(ghost_capacity, "ghost_capacity", allow_none=True)
    return TwoQueueGPUCachePolicy(ghost_factor=ghost_capacity)


def _build_dynamic(extra_config: dict[str, Any]) -> GPUCachePolicy:
    # Runtime-switchable LRU <-> S3-FIFO. `start_policy` picks the initial one;
    # the S3-FIFO knobs configure the inner S3 sub-policy (LRU takes none). The
    # `auto_control`/`control_*` keys configure the in-process control loop and
    # are consumed by the scheduler, not here — listed so they pass validation.
    known = {
        "start_policy",
        "ghost_factor",
        "promote_threshold",
        "freq_cap",
        "auto_control",
        "control_model",
        "control_kvsim_run",
        "control_interval_min",
        "control_margin",
        "control_consecutive",
        "control_min_requests",
        "control_dry_run",
    }
    unknown = set(extra_config) - known
    if unknown:
        raise ValueError(
            f"Unknown options for the 'dynamic' eviction policy: "
            f"{sorted(unknown)}. Supported options: {sorted(known)}."
        )
    start_policy = extra_config.get("start_policy", "lru")
    if start_policy not in ("lru", "s3_fifo"):
        raise ValueError(
            f"The 'dynamic' option 'start_policy' must be 'lru' or 's3_fifo', "
            f"got {start_policy!r}."
        )
    # Reuse the s3_fifo builder (and its validation) for the inner S3 knobs.
    s3_config = {
        key: extra_config[key]
        for key in ("ghost_factor", "promote_threshold", "freq_cap")
        if key in extra_config
    }
    lru = LRUGPUCachePolicy()
    s3 = _build_s3_fifo(s3_config)
    assert isinstance(s3, S3FIFOGPUCachePolicy)
    return DynamicGPUCachePolicy(lru=lru, s3=s3, start_policy=start_policy)


def _build_adaptive_main_gap(extra_config: dict[str, Any]) -> GPUCachePolicy:
    # Self-tuning LRU <-> S3-FIFO with an in-process analytical predictor. The
    # S3-FIFO knobs configure the inner S3 sub-policy; the adaptive knobs
    # configure the predictor + switch cadence.
    known = {
        "start_policy",
        "ghost_factor",
        "promote_threshold",
        "freq_cap",
        "margin",
        "interval_min",
        "min_references",
        "history_factor",
        "history_max",
        "consecutive",
    }
    unknown = set(extra_config) - known
    if unknown:
        raise ValueError(
            f"Unknown options for the 'adaptive_main_gap' eviction policy: "
            f"{sorted(unknown)}. Supported options: {sorted(known)}."
        )
    start_policy = extra_config.get("start_policy", "lru")
    if start_policy not in ("lru", "s3_fifo"):
        raise ValueError(
            f"The 'adaptive_main_gap' option 'start_policy' must be 'lru' or "
            f"'s3_fifo', got {start_policy!r}."
        )
    # Reuse the s3_fifo builder (and its validation) for the inner S3 knobs.
    s3_config = {
        key: extra_config[key]
        for key in ("ghost_factor", "promote_threshold", "freq_cap")
        if key in extra_config
    }
    lru = LRUGPUCachePolicy()
    s3 = _build_s3_fifo(s3_config)
    assert isinstance(s3, S3FIFOGPUCachePolicy)

    margin = extra_config.get("margin", DEFAULT_ADAPTIVE_MARGIN)
    _validate_nonneg_number(margin, "margin", policy="adaptive_main_gap")
    interval_min = extra_config.get("interval_min", DEFAULT_ADAPTIVE_INTERVAL_MIN)
    _validate_nonneg_number(interval_min, "interval_min", policy="adaptive_main_gap")
    if interval_min <= 0:
        raise ValueError(
            f"The 'adaptive_main_gap' option 'interval_min' must be > 0, "
            f"got {interval_min!r}."
        )
    min_references = extra_config.get(
        "min_references", DEFAULT_ADAPTIVE_MIN_REFERENCES
    )
    _validate_positive_int(
        min_references,
        "min_references",
        allow_none=False,
        policy="adaptive_main_gap",
    )
    history_factor = extra_config.get(
        "history_factor", DEFAULT_ADAPTIVE_HISTORY_FACTOR
    )
    _validate_nonneg_number(
        history_factor, "history_factor", policy="adaptive_main_gap"
    )
    if history_factor < 1.0:
        raise ValueError(
            f"The 'adaptive_main_gap' option 'history_factor' must be >= 1.0, "
            f"got {history_factor!r}."
        )
    history_max = extra_config.get("history_max", DEFAULT_ADAPTIVE_HISTORY_MAX)
    _validate_positive_int(
        history_max, "history_max", allow_none=True, policy="adaptive_main_gap"
    )
    consecutive = extra_config.get("consecutive", DEFAULT_ADAPTIVE_CONSECUTIVE)
    _validate_positive_int(
        consecutive, "consecutive", allow_none=False, policy="adaptive_main_gap"
    )
    return AdaptiveMainGapGPUCachePolicy(
        lru=lru,
        s3=s3,
        start_policy=start_policy,
        margin=float(margin),
        interval_min=float(interval_min),
        min_references=min_references,
        history_factor=float(history_factor),
        history_max=history_max,
        consecutive=consecutive,
    )


# Registry of policy builders. Each builder receives the policy's extra-config
# dict and reads the options it understands. Adding a policy (or an option to an
# existing one) never changes CacheConfig or any constructor signature.
_GPU_EVICTION_POLICIES: dict[str, Callable[[dict[str, Any]], GPUCachePolicy]] = {
    "lru": _build_lru,
    "lfu": _build_lfu,
    "priority": _build_priority,
    "s3_fifo": _build_s3_fifo,
    "2q": _build_twoqueue,
    "dynamic": _build_dynamic,
    "adaptive_main_gap": _build_adaptive_main_gap,
    "arc": _build_arc,
}


def make_gpu_eviction_policy(
    name: str, extra_config: dict[str, Any] | None = None
) -> GPUCachePolicy:
    """Build a GPU eviction policy.

    Args:
        name: Policy name (``"lru"``, ``"lfu"``, ``"priority"``, ``"s3_fifo"``,
            ``"2q"``, ``"dynamic"``, ``"adaptive_main_gap"``, or ``"arc"``).
        extra_config: Policy-specific options. For ``"lfu"``: ``max_freq``
            (int) and ``ghost_capacity`` (int | None). For ``"s3_fifo"``:
            ``ghost_factor`` (float >= 0, ghost size as a multiple of capacity),
            ``promote_threshold`` (int >= 1), and ``freq_cap`` (int >= 1). For
            ``"dynamic"``: ``start_policy`` (``"lru"`` | ``"s3_fifo"``, default
            ``"lru"``) plus the same S3-FIFO knobs for its inner S3 sub-policy.
            For ``"adaptive_main_gap"``: the same ``start_policy`` + S3-FIFO
            knobs, plus ``margin`` (float >= 0), ``interval_min`` (float > 0,
            wall-clock minutes between decisions), ``min_references`` (int >= 1,
            references a window must see to weigh a switch), ``history_factor``
            (float >= 1.0, tracked-hash budget as a multiple of capacity),
            ``history_max`` (int >= 1 or omitted; absolute ceiling on that
            budget, for very large pools), and ``consecutive`` (int >= 1,
            winning windows required to switch). ``"lru"`` and ``"priority"``
            take no options. Unknown keys raise.
    """
    try:
        builder = _GPU_EVICTION_POLICIES[name.lower()]
    except KeyError:
        valid = ", ".join(sorted(_GPU_EVICTION_POLICIES))
        raise ValueError(
            f"Unknown GPU eviction policy {name!r}. Valid options: {valid}."
        ) from None
    return builder(extra_config or {})
