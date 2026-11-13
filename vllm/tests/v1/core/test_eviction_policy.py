# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPU KV-cache eviction policies."""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.eviction_policy import (
    _HASHLESS_BUCKET,
    _S3_MAIN,
    _S3_NONE,
    _S3_SMALL,
    AdaptiveMainGapGPUCachePolicy,
    ARCGPUCachePolicy,
    DynamicGPUCachePolicy,
    LFUGPUCachePolicy,
    LRUGPUCachePolicy,
    PriorityGPUCachePolicy,
    S3FIFOGPUCachePolicy,
    _MainGapPredictor,
    make_gpu_eviction_policy,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)


def make_block(block_id: int, content: bytes | None = None) -> KVCacheBlock:
    """Build a block, optionally cached with the given content hash."""
    block = KVCacheBlock(block_id=block_id)
    if content is not None:
        block.set_block_hash(make_block_hash_with_group_id(BlockHash(content), 0))
    return block


def hit(policy: LFUGPUCachePolicy, block: KVCacheBlock) -> None:
    """Simulate a full cache-hit cycle: leave pool, count hit, return to pool."""
    policy.remove(block)
    policy.touch(block)
    policy.insert(block)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_factory_builds_known_policies():
    assert isinstance(make_gpu_eviction_policy("lru"), LRUGPUCachePolicy)
    assert isinstance(make_gpu_eviction_policy("lfu"), LFUGPUCachePolicy)
    assert isinstance(
        make_gpu_eviction_policy("priority"), PriorityGPUCachePolicy
    )
    # Case-insensitive.
    assert isinstance(make_gpu_eviction_policy("LFU"), LFUGPUCachePolicy)
    assert isinstance(
        make_gpu_eviction_policy("PRIORITY"), PriorityGPUCachePolicy
    )


def test_factory_rejects_unknown_policy():
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("nope")


def test_factory_reads_extra_config_for_lfu():
    lfu = make_gpu_eviction_policy("lfu", {"max_freq": 9, "ghost_capacity": 4})
    assert isinstance(lfu, LFUGPUCachePolicy)
    assert lfu._max_freq == 9
    assert lfu._ghost_capacity == 4


def test_factory_rejects_unknown_options():
    # Unknown key for the selected policy fails loudly (catches typos).
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("lfu", {"nope": 1})
    # lru accepts no options at all.
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("lru", {"max_freq": 9})
    # priority accepts no options either.
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("priority", {"max_freq": 9})


def test_factory_rejects_bad_lfu_params():
    # Out-of-range and wrong-type options fail with a clean ValueError (not a
    # stripped-under-`-O` AssertionError).
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("lfu", {"max_freq": 0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("lfu", {"ghost_capacity": 0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("lfu", {"max_freq": True})


# --------------------------------------------------------------------------- #
# LRU
# --------------------------------------------------------------------------- #
def test_lru_evicts_hashless_first_then_fifo():
    policy = make_gpu_eviction_policy("lru")
    cached = make_block(1, b"a")
    hashless = make_block(2)
    policy.insert_n([cached, hashless])
    # Hashless has no reuse value and is evicted before any cached block.
    assert policy.evict_n(1) == [hashless]
    assert policy.evict_n(1) == [cached]


def test_lru_touch_is_noop():
    policy = make_gpu_eviction_policy("lru")
    block = make_block(1, b"a")
    policy.insert(block)
    policy.touch(block)
    assert len(policy) == 1
    assert policy.evict_n(1) == [block]


def test_lru_remove_absent_block_is_noop():
    # The interface requires remove() to be a no-op on a non-candidate; the LRU
    # queue would otherwise raise on an unlinked block.
    policy = make_gpu_eviction_policy("lru")
    present = make_block(1, b"a")
    policy.insert(present)
    policy.remove(make_block(2, b"b"))  # never inserted; must not raise
    assert len(policy) == 1
    policy.remove(present)
    assert len(policy) == 0


def test_lru_iter_eviction_order_matches_get_all():
    policy = make_gpu_eviction_policy("lru")
    blocks = [make_block(i, bytes([i])) for i in (1, 2, 3)]
    policy.insert_n(blocks)
    assert list(policy.iter_eviction_order()) == policy.get_all_free_blocks()


def test_lru_iter_eviction_order_resumes_after_cursor():
    policy = make_gpu_eviction_policy("lru")
    blocks = [make_block(i, bytes([i])) for i in (1, 2, 3)]
    policy.insert_n(blocks)
    order = policy.get_all_free_blocks()
    # Resuming after the first candidate yields only the rest.
    assert list(policy.iter_eviction_order(start_after=order[0])) == order[1:]


def test_lru_iter_eviction_order_restarts_when_cursor_evicted():
    policy = make_gpu_eviction_policy("lru")
    blocks = [make_block(i, bytes([i])) for i in (1, 2, 3)]
    policy.insert_n(blocks)
    stale = policy.evict_n(1)[0]  # leaves the pool -> cursor no longer valid
    # A stale cursor restarts from the current best candidate.
    assert list(policy.iter_eviction_order(start_after=stale)) == (
        policy.get_all_free_blocks()
    )


# --------------------------------------------------------------------------- #
# LFU
# --------------------------------------------------------------------------- #
def test_lfu_evicts_hashless_before_cached():
    policy = LFUGPUCachePolicy()
    cached = make_block(1, b"a")
    hashless = make_block(2)
    policy.insert_n([cached, hashless])
    assert policy.evict_n(1) == [hashless]
    assert policy.evict_n(1) == [cached]


def test_lfu_lru_tiebreak_within_frequency():
    policy = LFUGPUCachePolicy()
    blocks = [make_block(i, bytes([i])) for i in (1, 2, 3)]
    policy.insert_n(blocks)  # all fresh -> frequency 1, insertion-ordered
    # Same frequency: the least recently inserted goes first.
    assert policy.evict_n(2) == [blocks[0], blocks[1]]


def test_lfu_touch_raises_frequency():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    policy.insert_n([b1, b2])
    hit(policy, b1)  # frequency(a) -> 2
    # b2 is now the least frequently used.
    assert policy.evict_n(1) == [b2]
    assert policy.evict_n(1) == [b1]


def test_lfu_evict_spans_buckets():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    policy.insert_n([b1, b2])
    hit(policy, b2)  # b2 -> frequency 2
    assert policy.evict_n(2) == [b1, b2]  # low frequency first
    assert len(policy) == 0


def test_lfu_ghost_frequency_survives_eviction_and_resurrects():
    policy = LFUGPUCachePolicy()
    hot = make_block(1, b"hot")
    policy.insert(hot)
    for _ in range(3):
        hit(policy, hot)  # frequency(hot) -> 4
    # Evict it: the physical block is reused, but the content's frequency is
    # a ghost that must persist.
    assert policy.evict_n(1) == [hot]

    # The same content is later cached into a *different* physical block.
    hot_again = make_block(2, b"hot")
    cold = make_block(3, b"cold")
    policy.insert_n([hot_again, cold])
    # hot_again resurrects at the ghost frequency (4), so the fresh cold block
    # (frequency 1) is evicted first.
    assert policy.evict_n(1) == [cold]
    assert policy.evict_n(1) == [hot_again]


def test_lfu_remove_absent_block_is_noop():
    policy = LFUGPUCachePolicy()
    present = make_block(1, b"a")
    policy.insert(present)
    policy.remove(make_block(2, b"b"))  # never inserted; must not raise
    assert len(policy) == 1
    policy.remove(present)
    assert len(policy) == 0


def test_lfu_handles_duplicate_content_hashes():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"dup")
    b2 = make_block(2, b"dup")  # same content, distinct physical block
    policy.insert_n([b1, b2])
    assert len(policy) == 2
    hit(policy, b1)  # advances frequency(dup); b2 stays where it was
    assert policy.evict_n(1) == [b2]
    assert policy.evict_n(1) == [b1]
    assert len(policy) == 0


def test_lfu_records_bucket_on_block():
    policy = LFUGPUCachePolicy()
    block = make_block(1, b"a")
    assert block.eviction_bucket == -1  # not yet a candidate
    policy.insert(block)
    assert block.eviction_bucket == 1  # cached content, frequency 1
    assert policy.evict_n(1) == [block]
    assert block.eviction_bucket == -1  # reset when it leaves the pool


def test_lfu_evict_more_than_available_raises():
    policy = LFUGPUCachePolicy()
    policy.insert(make_block(1, b"a"))
    with pytest.raises(ValueError):
        policy.evict_n(2)


def test_lfu_frequency_saturates_at_max():
    policy = LFUGPUCachePolicy(max_freq=2)
    block = make_block(1, b"a")
    policy.insert(block)
    for _ in range(5):
        policy.touch(block)  # capped at max_freq
    policy.remove(block)
    policy.insert(block)
    assert policy._freq[block.block_hash] == 2


def test_lfu_reset_forgets_frequencies():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    policy.insert_n([b1, b2])
    hit(policy, b1)  # give "a" a higher frequency
    # BlockPool clears every block hash before calling reset(); mirror that.
    b1.reset_hash()
    b2.reset_hash()
    policy.reset()
    assert len(policy) == 2
    assert dict(policy._freq) == {}  # popularity forgotten
    assert set(policy._buckets) == {0}  # all candidates now hashless


def test_lfu_ghost_capacity_trims_least_recently_used():
    policy = LFUGPUCachePolicy(ghost_capacity=2)
    for i, content in enumerate([b"a", b"b", b"c"], start=1):
        block = make_block(i, content)
        policy.insert(block)
        hit(policy, block)
    # Only the two most recently touched ghost entries survive.
    assert len(policy._freq) == 2
    evicted = make_block_hash_with_group_id(BlockHash(b"a"), 0)
    assert evicted not in policy._freq


def test_lfu_reinsert_refreshes_ghost_recency():
    policy = LFUGPUCachePolicy(ghost_capacity=2)
    a = make_block(1, b"a")
    b = make_block(2, b"b")
    policy.insert_n([a, b])  # ghost recency order: a (older), b (newer)
    # Re-inserting "a" (a resident cache-hit cycle) must refresh its recency,
    # so the next fresh entry trims "b" (now coldest), not "a".
    policy.remove(a)
    policy.insert(a)
    policy.insert(make_block(3, b"c"))
    assert make_block_hash_with_group_id(BlockHash(b"b"), 0) not in policy._freq
    assert make_block_hash_with_group_id(BlockHash(b"a"), 0) in policy._freq


def test_lfu_get_all_free_blocks_in_eviction_order():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    hashless = make_block(3)
    policy.insert_n([b1, b2, hashless])
    hit(policy, b1)  # b1 -> frequency 2
    # Hashless (bucket 0) first, then frequency 1 (b2), then frequency 2 (b1).
    assert policy.get_all_free_blocks() == [hashless, b2, b1]


def test_lfu_iter_eviction_order_matches_get_all():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    hashless = make_block(3)
    policy.insert_n([b1, b2, hashless])
    hit(policy, b1)  # b1 -> frequency 2
    assert list(policy.iter_eviction_order()) == [hashless, b2, b1]


def test_lfu_iter_eviction_order_resumes_across_buckets():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    hashless = make_block(3)
    policy.insert_n([b1, b2, hashless])
    hit(policy, b1)  # eviction order: [hashless (0), b2 (1), b1 (2)]
    # Resume past a bucket boundary in both directions.
    assert list(policy.iter_eviction_order(start_after=hashless)) == [b2, b1]
    assert list(policy.iter_eviction_order(start_after=b2)) == [b1]


def test_lfu_iter_eviction_order_restarts_when_cursor_evicted():
    policy = LFUGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    policy.insert_n([b1, b2])
    stale = policy.evict_n(1)[0]  # b1 leaves the pool -> cursor invalid
    assert list(policy.iter_eviction_order(start_after=stale)) == (
        policy.get_all_free_blocks()
    )


# --------------------------------------------------------------------------- #
# Priority
# --------------------------------------------------------------------------- #
def cache_at(
    policy: PriorityGPUCachePolicy, block: KVCacheBlock, priority: int
) -> None:
    """Simulate BlockPool caching a fresh block at the given priority."""
    policy.on_cache(block, priority)
    policy.insert(block)


def phit(
    policy: PriorityGPUCachePolicy, block: KVCacheBlock, priority: int
) -> None:
    """Simulate a cache-hit cycle by a request of the given priority."""
    policy.remove(block)
    policy.touch(block, priority)
    policy.insert(block)


def test_priority_lower_number_evicted_last():
    policy = PriorityGPUCachePolicy()
    important = make_block(1, b"a")
    trivial = make_block(2, b"b")
    cache_at(policy, important, 0)  # lower = more important
    cache_at(policy, trivial, 2)
    # Least important (highest number) is evicted first.
    assert policy.evict_n(1) == [trivial]
    assert policy.evict_n(1) == [important]


def test_priority_hashless_evicted_before_cached():
    policy = PriorityGPUCachePolicy()
    cached = make_block(1, b"a")
    hashless = make_block(2)
    cache_at(policy, cached, 5)  # even low-importance cached beats hashless
    policy.insert(hashless)
    assert policy.evict_n(1) == [hashless]
    assert policy.evict_n(1) == [cached]


def test_priority_lru_tiebreak_within_priority():
    policy = PriorityGPUCachePolicy()
    blocks = [make_block(i, bytes([i])) for i in (1, 2, 3)]
    for block in blocks:
        cache_at(policy, block, 1)  # same priority, insertion-ordered
    # Same priority: the least recently freed goes first.
    assert policy.evict_n(2) == [blocks[0], blocks[1]]


def test_priority_shared_block_keeps_best_priority():
    # A block used by both a low- and a high-priority request must be protected
    # at the high-priority (lowest-number) level, regardless of order.
    policy = PriorityGPUCachePolicy()
    shared = make_block(1, b"a")
    policy.on_cache(shared, 5)  # first cached by a low-priority request
    policy.on_cache(shared, 0)  # then a high-priority request also uses it
    policy.insert(shared)
    other = make_block(2, b"b")
    cache_at(policy, other, 3)
    assert shared.eviction_priority == 0  # best (lowest) wins
    assert policy.evict_n(1) == [other]  # priority 3 goes before priority 0
    assert policy.evict_n(1) == [shared]


def test_priority_shared_via_touch():
    # The high-priority contribution can arrive as a later cache hit.
    policy = PriorityGPUCachePolicy()
    shared = make_block(1, b"a")
    cache_at(policy, shared, 4)
    phit(policy, shared, 1)  # a higher-priority request hits it
    assert shared.eviction_priority == 1
    low = make_block(2, b"b")
    cache_at(policy, low, 2)
    assert policy.evict_n(1) == [low]
    assert policy.evict_n(1) == [shared]


def test_priority_touch_clamps_negative_to_top_tier():
    policy = PriorityGPUCachePolicy()
    block = make_block(1, b"a")
    cache_at(policy, block, -5)  # negative clamps to 0 (cannot beat top tier)
    assert block.eviction_priority == 0


def test_priority_replicas_are_independent():
    # Distinct physical blocks sharing a content hash carry their own priority
    # (there is no cross-block ghost) and are evicted independently.
    policy = PriorityGPUCachePolicy()
    b1 = make_block(1, b"dup")
    b2 = make_block(2, b"dup")  # same content, distinct physical block
    cache_at(policy, b1, 0)
    cache_at(policy, b2, 3)
    assert policy.evict_n(1) == [b2]  # its own priority (3), evicted first
    assert policy.evict_n(1) == [b1]


def test_priority_no_persistence_across_eviction():
    policy = PriorityGPUCachePolicy()
    block = make_block(1, b"a")
    cache_at(policy, block, 0)
    assert block.eviction_priority == 0
    assert policy.evict_n(1) == [block]
    # Reuse resets the block's priority: importance is not persisted.
    assert block.eviction_priority == -1
    assert block.eviction_bucket == -1

    # The same content re-cached by a low-priority request does NOT resurrect
    # its former top priority.
    block.reset_hash()
    block.set_block_hash(make_block_hash_with_group_id(BlockHash(b"a"), 0))
    cache_at(policy, block, 5)
    cold = make_block(2, b"x")
    cache_at(policy, cold, 0)
    assert policy.evict_n(1) == [block]  # now priority 5, evicted before cold


def test_priority_unknown_priority_is_least_important():
    # A cached block that never received a priority (defensive path) is treated
    # as least important, evicted before any known-priority content.
    policy = PriorityGPUCachePolicy()
    known = make_block(1, b"a")
    cache_at(policy, known, 2)  # low importance, but known
    unknown = make_block(2, b"b")
    policy.insert(unknown)  # cached but no on_cache/touch -> priority unset
    assert policy.evict_n(1) == [unknown]
    assert policy.evict_n(1) == [known]


def test_priority_records_bucket_and_priority_on_block():
    policy = PriorityGPUCachePolicy()
    block = make_block(1, b"a")
    assert block.eviction_bucket == -1
    assert block.eviction_priority == -1
    cache_at(policy, block, 2)
    assert block.eviction_bucket == 2
    assert block.eviction_priority == 2
    hashless = make_block(2)
    policy.insert(hashless)
    assert hashless.eviction_bucket == _HASHLESS_BUCKET
    assert policy.evict_n(2)
    assert block.eviction_bucket == -1


def test_priority_remove_absent_block_is_noop():
    policy = PriorityGPUCachePolicy()
    present = make_block(1, b"a")
    cache_at(policy, present, 0)
    policy.remove(make_block(2, b"b"))  # never inserted; must not raise
    assert len(policy) == 1
    policy.remove(present)
    assert len(policy) == 0


def test_priority_evict_more_than_available_raises():
    policy = PriorityGPUCachePolicy()
    cache_at(policy, make_block(1, b"a"), 0)
    with pytest.raises(ValueError):
        policy.evict_n(2)


def test_priority_reset_forgets_priorities():
    policy = PriorityGPUCachePolicy()
    b1 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    cache_at(policy, b1, 0)
    cache_at(policy, b2, 3)
    # BlockPool clears every block hash before calling reset(); mirror that.
    b1.reset_hash()
    b2.reset_hash()
    policy.reset()
    assert len(policy) == 2
    assert b1.eviction_priority == -1
    assert b2.eviction_priority == -1
    assert not policy._buckets  # all candidates now hashless


def test_priority_get_all_free_blocks_in_eviction_order():
    policy = PriorityGPUCachePolicy()
    b0 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    hashless = make_block(3)
    cache_at(policy, b0, 0)
    cache_at(policy, b2, 2)
    policy.insert(hashless)
    # Hashless first, then priority 2 (least important), then priority 0.
    assert policy.get_all_free_blocks() == [hashless, b2, b0]


def test_priority_iter_eviction_order_matches_get_all():
    policy = PriorityGPUCachePolicy()
    b0 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    hashless = make_block(3)
    cache_at(policy, b0, 0)
    cache_at(policy, b2, 2)
    policy.insert(hashless)
    assert list(policy.iter_eviction_order()) == [hashless, b2, b0]


def test_priority_iter_eviction_order_resumes_across_buckets():
    policy = PriorityGPUCachePolicy()
    b0 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    hashless = make_block(3)
    cache_at(policy, b0, 0)
    cache_at(policy, b2, 2)
    policy.insert(hashless)
    # Order: [hashless, b2 (prio 2), b0 (prio 0)]; resume past each boundary.
    assert list(policy.iter_eviction_order(start_after=hashless)) == [b2, b0]
    assert list(policy.iter_eviction_order(start_after=b2)) == [b0]


def test_priority_iter_eviction_order_restarts_when_cursor_evicted():
    policy = PriorityGPUCachePolicy()
    b0 = make_block(1, b"a")
    b2 = make_block(2, b"b")
    cache_at(policy, b0, 0)
    cache_at(policy, b2, 2)
    stale = policy.evict_n(1)[0]  # b2 leaves the pool -> cursor invalid
    assert list(policy.iter_eviction_order(start_after=stale)) == (
        policy.get_all_free_blocks()
    )


# --------------------------------------------------------------------------- #
# BlockPool integration
# --------------------------------------------------------------------------- #
def test_block_pool_lfu_alloc_free_roundtrip():
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="lfu",
    )
    # One block is reserved as the null block.
    assert pool.get_num_free_blocks() == 7
    blocks = pool.get_new_blocks(3)
    assert len(blocks) == 3
    assert pool.get_num_free_blocks() == 4
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 7


def test_block_pool_lfu_hides_free_block_queue():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="lfu",
    )
    # The LRU linked list is not available under non-LRU policies.
    with pytest.raises(NotImplementedError):
        _ = pool.free_block_queue


def test_block_pool_lru_exposes_free_block_queue():
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    assert pool.free_block_queue.num_free_blocks == 3


def test_block_pool_forwards_eviction_extra_config():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="lfu",
        eviction_policy_extra_config={"max_freq": 7},
    )
    assert isinstance(pool._policy, LFUGPUCachePolicy)
    assert pool._policy._max_freq == 7


def test_block_pool_priority_alloc_free_roundtrip():
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="priority",
    )
    assert isinstance(pool._policy, PriorityGPUCachePolicy)
    # One block is reserved as the null block.
    assert pool.get_num_free_blocks() == 7
    blocks = pool.get_new_blocks(3)
    assert len(blocks) == 3
    assert pool.get_num_free_blocks() == 4
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 7


def test_block_pool_priority_hides_free_block_queue():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="priority",
    )
    # The LRU linked list is not available under non-LRU policies.
    with pytest.raises(NotImplementedError):
        _ = pool.free_block_queue


# --------------------------------------------------------------------------- #
# S3-FIFO
# --------------------------------------------------------------------------- #
def make_s3(capacity: int = 100, **extra) -> S3FIFOGPUCachePolicy:
    """Build an S3-FIFO policy and size it from the given pool capacity."""
    policy = make_gpu_eviction_policy("s3_fifo", extra)
    assert isinstance(policy, S3FIFOGPUCachePolicy)
    policy.set_pool_capacity(capacity)
    return policy


def s3_cache(policy: S3FIFOGPUCachePolicy, block: KVCacheBlock) -> None:
    """Simulate BlockPool caching a fresh block, then freeing it."""
    policy.on_cache(block, 0)
    policy.insert(block)


def s3_hit(policy: S3FIFOGPUCachePolicy, block: KVCacheBlock) -> None:
    """Simulate a cache-hit cycle: leave pool, count hit, return to pool."""
    policy.remove(block)
    policy.touch(block)
    policy.insert(block)


def s3_ghost_key(content: bytes):
    return make_block_hash_with_group_id(BlockHash(content), 0)


def test_factory_builds_s3_fifo():
    assert isinstance(make_gpu_eviction_policy("s3_fifo"), S3FIFOGPUCachePolicy)
    # Case-insensitive.
    assert isinstance(make_gpu_eviction_policy("S3_FIFO"), S3FIFOGPUCachePolicy)


def test_factory_reads_extra_config_for_s3_fifo():
    policy = make_gpu_eviction_policy(
        "s3_fifo",
        {"ghost_factor": 2.0, "promote_threshold": 2, "freq_cap": 5},
    )
    assert isinstance(policy, S3FIFOGPUCachePolicy)
    assert policy._ghost_factor == 2.0
    assert policy._promote_threshold == 2
    assert policy._freq_cap == 5


def test_factory_rejects_bad_s3_params():
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("s3_fifo", {"ghost_factor": -1})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("s3_fifo", {"promote_threshold": 0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("s3_fifo", {"freq_cap": True})  # bool rejected
    with pytest.raises(ValueError):
        # promote_threshold must not exceed freq_cap.
        make_gpu_eviction_policy("s3_fifo", {"promote_threshold": 5, "freq_cap": 3})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("s3_fifo", {"nope": 1})


def test_s3_set_pool_capacity_sizes_ghost():
    policy = make_gpu_eviction_policy("s3_fifo")  # default ghost_factor 1.0
    assert isinstance(policy, S3FIFOGPUCachePolicy)
    policy.set_pool_capacity(100)
    assert policy._ghost_capacity == 100  # round(1.0 * 100)


def test_s3_new_content_enters_small_hashless_evicted_first():
    policy = make_s3()
    cached = make_block(1, b"a")
    hashless = make_block(2)
    s3_cache(policy, cached)  # -> small
    policy.insert(hashless)  # -> hashless
    assert cached.s3_queue == _S3_SMALL
    # Hashless has no reuse value: evicted before any cached block.
    assert policy.evict_n(1) == [hashless]
    assert policy.evict_n(1) == [cached]


def test_s3_cold_small_block_demoted_to_ghost():
    policy = make_s3()
    x = make_block(1, b"x")
    s3_cache(policy, x)  # small, frequency 0
    assert policy.evict_n(1) == [x]  # cold -> demoted (fingerprint kept)
    assert s3_ghost_key(b"x") in policy._ghost
    assert x.s3_queue == _S3_NONE and x.s3_freq == 0


def test_s3_ghost_hit_admits_to_main():
    policy = make_s3()
    x = make_block(1, b"x")
    s3_cache(policy, x)
    assert policy.evict_n(1) == [x]  # demote to ghost
    # The same content, cached into a different physical block, is a ghost hit.
    x2 = make_block(2, b"x")
    policy.on_cache(x2, 0)
    assert x2.s3_queue == _S3_MAIN
    assert s3_ghost_key(b"x") not in policy._ghost  # consumed by the promotion


def test_s3_hit_promotes_small_to_main():
    policy = make_s3()
    hot = make_block(1, b"hot")
    cold = make_block(2, b"cold")
    s3_cache(policy, hot)  # small
    s3_hit(policy, hot)  # crosses promote_threshold (1) -> relabeled main
    assert hot.s3_queue == _S3_MAIN
    assert hot.s3_freq == 0  # reset on promotion
    s3_cache(policy, cold)  # small
    # Small-first eviction takes the cold small block; the promoted one is safe.
    assert policy.evict_n(1) == [cold]
    assert hot.s3_queue == _S3_MAIN


def test_s3_main_second_chance():
    policy = make_s3()
    a = make_block(1, b"a")
    b = make_block(2, b"b")
    # Force both into main via a pre-seeded ghost, then free them.
    policy._ghost[s3_ghost_key(b"a")] = None
    policy._ghost[s3_ghost_key(b"b")] = None
    policy.on_cache(a, 0)
    policy.insert(a)  # main: [a]
    policy.on_cache(b, 0)
    policy.insert(b)  # main: [a, b]
    a.s3_freq = 1  # a is warm
    # Main eviction: a (freq 1) gets a second chance, b (freq 0) is evicted.
    assert policy.evict_n(1) == [b]
    assert a.s3_freq == 0
    assert a.s3_queue == _S3_MAIN


def test_s3_evicts_small_before_main():
    # With no size target, a free small (probationary) block is always reused
    # before a main (protected) block.
    policy = make_s3()
    m = make_block(1, b"m")
    policy._ghost[s3_ghost_key(b"m")] = None
    policy.on_cache(m, 0)
    policy.insert(m)  # ghost hit -> main
    s = make_block(2, b"s")
    s3_cache(policy, s)  # small
    assert policy.evict_n(1) == [s]  # small before main
    assert m.s3_queue == _S3_MAIN  # main protected


def test_s3_reaccessed_blocks_live_in_main():
    # Reaccessed blocks are eager-promoted to main on their qualifying hit, so
    # small-eviction finds nothing and main-eviction returns the oldest.
    policy = make_s3()
    a = make_block(1, b"a")
    b = make_block(2, b"b")
    s3_cache(policy, a)
    s3_hit(policy, a)  # a -> main
    s3_cache(policy, b)
    s3_hit(policy, b)  # b -> main
    assert a.s3_queue == _S3_MAIN and b.s3_queue == _S3_MAIN
    assert policy.evict_n(1) == [a]  # oldest in main
    assert b.s3_queue == _S3_MAIN


def test_s3_replicas_are_independent():
    # Distinct physical blocks sharing a content hash carry their own frequency
    # and queue label; only the ghost is content-keyed.
    policy = make_s3(promote_threshold=2)  # one hit stays below the threshold
    b1 = make_block(1, b"dup")
    b2 = make_block(2, b"dup")  # same content, distinct physical block
    s3_cache(policy, b1)
    s3_cache(policy, b2)
    assert b1.s3_queue == _S3_SMALL and b2.s3_queue == _S3_SMALL
    s3_hit(policy, b1)  # only b1's frequency advances (1 < threshold 2)
    assert b1.s3_freq == 1 and b1.s3_queue == _S3_SMALL
    assert b2.s3_freq == 0 and b2.s3_queue == _S3_SMALL


def test_s3_invalidated_block_becomes_hashless():
    # Connector force-eviction can reset a block's hash while it is resident;
    # when it is later freed as hashless, insert clears the stale label.
    policy = make_s3()
    blk = make_block(1, b"a")
    policy.on_cache(blk, 0)  # labeled small while "in use" (not queued)
    assert blk.s3_queue == _S3_SMALL
    blk.reset_hash()
    policy.insert(blk)
    assert blk.s3_queue == _S3_NONE
    assert len(policy) == 1  # now sits in the hashless queue
    assert policy.evict_n(1) == [blk]


def test_s3_remove_absent_block_is_noop():
    policy = make_s3()
    present = make_block(1, b"a")
    s3_cache(policy, present)
    policy.remove(make_block(2, b"b"))  # never inserted; must not raise
    assert len(policy) == 1
    policy.remove(present)
    assert len(policy) == 0


def test_s3_evict_more_than_available_raises():
    policy = make_s3()
    s3_cache(policy, make_block(1, b"a"))
    with pytest.raises(ValueError):
        policy.evict_n(2)


def test_s3_reset_forgets_everything():
    policy = make_s3()
    a = make_block(1, b"a")
    b = make_block(2, b"b")
    s3_cache(policy, a)
    s3_cache(policy, b)
    s3_hit(policy, a)
    # BlockPool clears every block hash before calling reset(); mirror that.
    a.reset_hash()
    b.reset_hash()
    policy.reset()
    assert len(policy) == 2
    assert len(policy._ghost) == 0
    assert a.s3_queue == _S3_NONE and a.s3_freq == 0


def test_s3_get_all_free_blocks_in_eviction_order():
    policy = make_s3()
    m = make_block(1, b"m")
    s = make_block(2, b"s")
    hashless = make_block(3)
    policy._ghost[s3_ghost_key(b"m")] = None
    policy.on_cache(m, 0)
    policy.insert(m)  # main
    s3_cache(policy, s)  # small
    policy.insert(hashless)  # hashless
    # Order: hashless, then small, then main.
    assert policy.get_all_free_blocks() == [hashless, s, m]
    assert list(policy.iter_eviction_order()) == [hashless, s, m]


def test_s3_iter_eviction_order_resumes_and_restarts():
    policy = make_s3()
    m = make_block(1, b"m")
    s = make_block(2, b"s")
    hashless = make_block(3)
    policy._ghost[s3_ghost_key(b"m")] = None
    policy.on_cache(m, 0)
    policy.insert(m)
    s3_cache(policy, s)
    policy.insert(hashless)
    order = policy.get_all_free_blocks()  # [hashless, s, m]
    assert list(policy.iter_eviction_order(start_after=order[0])) == order[1:]
    assert list(policy.iter_eviction_order(start_after=order[1])) == order[2:]
    # A stale cursor (evicted block) restarts from the current best candidate.
    stale = policy.evict_n(1)[0]  # hashless leaves the pool
    assert list(policy.iter_eviction_order(start_after=stale)) == (
        policy.get_all_free_blocks()
    )


def test_block_pool_s3_fifo_alloc_free_roundtrip():
    pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="s3_fifo",
    )
    assert isinstance(pool._policy, S3FIFOGPUCachePolicy)
    # One block is reserved as the null block.
    assert pool.get_num_free_blocks() == 7
    blocks = pool.get_new_blocks(3)
    assert len(blocks) == 3
    assert pool.get_num_free_blocks() == 4
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 7


def test_block_pool_s3_fifo_sets_capacity_from_pool():
    pool = BlockPool(
        num_gpu_blocks=100,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="s3_fifo",
    )
    assert isinstance(pool._policy, S3FIFOGPUCachePolicy)
    assert pool._policy._ghost_capacity == 100  # round(1.0 * 100)


def test_block_pool_s3_fifo_hides_free_block_queue():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="s3_fifo",
    )
    # The LRU linked list is not available under non-LRU policies.
    with pytest.raises(NotImplementedError):
        _ = pool.free_block_queue


def test_block_pool_forwards_s3_extra_config():
    pool = BlockPool(
        num_gpu_blocks=100,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="s3_fifo",
        eviction_policy_extra_config={"ghost_factor": 2.0},
    )
    assert isinstance(pool._policy, S3FIFOGPUCachePolicy)
    assert pool._policy._ghost_factor == 2.0
    assert pool._policy._ghost_capacity == 200  # round(2.0 * 100)


# --------------------------------------------------------------------------- #
# Dynamic (runtime-switchable LRU <-> S3-FIFO)
# --------------------------------------------------------------------------- #
def make_dynamic(capacity: int = 100, **extra) -> DynamicGPUCachePolicy:
    """Build a dynamic policy and size it from the given pool capacity."""
    policy = make_gpu_eviction_policy("dynamic", extra)
    assert isinstance(policy, DynamicGPUCachePolicy)
    policy.set_pool_capacity(capacity)
    return policy


def free_ids(policy: DynamicGPUCachePolicy) -> set[int]:
    return {b.block_id for b in policy.get_all_free_blocks()}


def test_factory_builds_dynamic():
    policy = make_gpu_eviction_policy("dynamic")
    assert isinstance(policy, DynamicGPUCachePolicy)
    assert policy.active_policy == "lru"
    # Case-insensitive.
    assert isinstance(make_gpu_eviction_policy("DYNAMIC"), DynamicGPUCachePolicy)


def test_factory_reads_extra_config_for_dynamic():
    policy = make_gpu_eviction_policy(
        "dynamic",
        {"start_policy": "s3_fifo", "promote_threshold": 2, "freq_cap": 5},
    )
    assert isinstance(policy, DynamicGPUCachePolicy)
    assert policy.active_policy == "s3_fifo"
    # S3 knobs forwarded to the inner S3 sub-policy.
    assert policy._s3._promote_threshold == 2
    assert policy._s3._freq_cap == 5


def test_factory_rejects_bad_dynamic_params():
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("dynamic", {"start_policy": "lfu"})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("dynamic", {"nope": 1})
    # Inner S3 validation still applies.
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("dynamic", {"promote_threshold": 5, "freq_cap": 3})


def test_dynamic_delegates_to_active_lru():
    policy = make_dynamic()
    blocks = [make_block(i, str(i).encode()) for i in range(5)]
    policy.insert_n(blocks)
    assert len(policy) == 5
    # dyn_clock stamped increasing in list order (blocks[0] freed "first").
    assert [b.dyn_clock for b in blocks] == sorted(b.dyn_clock for b in blocks)
    # Eviction order comes from the active LRU sub-policy.
    assert [b.block_id for b in policy.evict_n(2)] == [0, 1]


def test_dynamic_tracks_universal_stats_regardless_of_active():
    policy = make_dynamic()  # LRU active: touch is a no-op there
    block = make_block(0, b"x")
    policy.insert(block)
    assert block.dyn_freq == 0
    policy.touch(block)
    policy.touch(block)
    assert block.dyn_freq == 2  # tracked by the wrapper even under LRU
    policy.on_cache(block, 0)
    assert block.dyn_freq == 0  # fresh content restarts the count
    # Hashless blocks never accrue frequency.
    hashless = make_block(1)
    policy.insert(hashless)
    policy.touch(hashless)
    assert hashless.dyn_freq == 0


def test_dynamic_switch_lru_to_s3_preserves_set_and_partitions():
    policy = make_dynamic(promote_threshold=2, freq_cap=3)
    b0 = make_block(0, b"a")
    b1 = make_block(1, b"b")
    b2 = make_block(2, b"c")
    b3 = make_block(3, b"d")
    hashless = make_block(4)
    for blk in (b0, b1, b2, b3, hashless):
        policy.insert(blk)
    policy.touch(b1)
    policy.touch(b1)  # dyn_freq 2 >= threshold -> main
    policy.touch(b2)  # dyn_freq 1 < threshold  -> small
    before = free_ids(policy)

    assert policy.set_active_policy("s3_fifo") is True
    assert policy.active_policy == "s3_fifo"
    assert len(policy) == 5
    assert free_ids(policy) == before
    assert len(policy._lru) == 0  # source drained
    # Partition derived from dyn_freq; s3_freq seeded 0 (not from dyn_freq).
    assert b1.s3_queue == _S3_MAIN
    assert b0.s3_queue == _S3_SMALL
    assert b2.s3_queue == _S3_SMALL
    assert b3.s3_queue == _S3_SMALL
    assert hashless.s3_queue == _S3_NONE
    assert all(b.s3_freq == 0 for b in (b0, b1, b2, b3))
    # Eviction under S3: hashless, then small (FIFO by free-time), then main.
    assert [b.block_id for b in policy.evict_n(5)] == [4, 0, 2, 3, 1]


def test_dynamic_switch_s3_to_lru_rebuilds_recency():
    policy = make_dynamic(start_policy="s3_fifo")
    cached = [make_block(i, str(i).encode()) for i in range(4)]
    for b in cached:
        policy.insert(b)
    hashless = make_block(9)
    policy.insert(hashless)
    before = free_ids(policy)

    assert policy.set_active_policy("lru") is True
    assert len(policy._s3) == 0  # source drained
    assert free_ids(policy) == before
    # LRU order after rebuild: hashless first, then by dyn_clock ascending.
    assert [b.block_id for b in policy.evict_n(5)] == [9, 0, 1, 2, 3]


def test_dynamic_switch_to_current_is_noop():
    policy = make_dynamic()
    assert policy.set_active_policy("lru") is False
    assert policy.active_policy == "lru"


def test_dynamic_round_trip_preserves_free_set():
    policy = make_dynamic()
    blocks = [make_block(i, str(i).encode()) for i in range(6)]
    policy.insert_n(blocks)
    before = free_ids(policy)
    policy.set_active_policy("s3_fifo")
    policy.set_active_policy("lru")
    assert policy.active_policy == "lru"
    assert free_ids(policy) == before
    assert len(policy) == 6
    assert len(policy._s3) == 0


def test_dynamic_set_active_policy_rejects_unknown():
    policy = make_dynamic()
    with pytest.raises(ValueError):
        policy.set_active_policy("lfu")


def test_dynamic_switch_ignores_in_use_blocks():
    policy = make_dynamic()
    blocks = [make_block(i, str(i).encode()) for i in range(5)]
    policy.insert_n(blocks)
    policy.remove(blocks[1])  # simulate going in-use
    policy.remove(blocks[3])
    assert len(policy) == 3
    policy.set_active_policy("s3_fifo")
    assert len(policy) == 3
    assert free_ids(policy) == {0, 2, 4}
    # In-use blocks stay unlinked and are not migrated.
    assert blocks[1].prev_free_block is None
    assert blocks[1].next_free_block is None


def test_block_pool_selects_dynamic_policy():
    pool = BlockPool(
        num_gpu_blocks=100,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="dynamic",
        eviction_policy_extra_config={"start_policy": "s3_fifo"},
    )
    assert isinstance(pool._policy, DynamicGPUCachePolicy)
    assert pool._policy.active_policy == "s3_fifo"
    # set_pool_capacity is forwarded to the inner S3 (ghost sized from capacity).
    assert pool._policy._s3._ghost_capacity == 100


def test_block_pool_set_eviction_policy_switches():
    pool = BlockPool(
        num_gpu_blocks=100,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="dynamic",
    )
    assert pool.get_active_eviction_policy() == "lru"
    assert pool.set_eviction_policy("s3_fifo") is True
    assert pool.get_active_eviction_policy() == "s3_fifo"
    # Switching to the current policy is a no-op.
    assert pool.set_eviction_policy("s3_fifo") is False


def test_block_pool_set_eviction_policy_rejects_non_dynamic():
    pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="lru",
    )
    assert pool.get_active_eviction_policy() is None
    with pytest.raises(RuntimeError):
        pool.set_eviction_policy("s3_fifo")


# --------------------------------------------------------------------------- #
# Adaptive main-gap predictor (analytical LRU-vs-S3-FIFO estimator)
# --------------------------------------------------------------------------- #
def bh(content: bytes):
    """A content hash key for the predictor (group 0)."""
    return make_block_hash_with_group_id(BlockHash(content), 0)


def feed(predictor: _MainGapPredictor, seq: list[bytes]) -> None:
    for content in seq:
        predictor.record_ref(bh(content))


def test_predictor_neutral_before_any_reuse():
    # With no reuse, both estimates are 0 and there is nothing to decide on.
    p = _MainGapPredictor()
    p.set_pool_capacity(10)
    feed(p, [b"a", b"b", b"c"])  # three first-time refs, no reuse
    assert p._win_reuse == 0
    assert p.decide() == (0.0, 0.0)


def test_predictor_s3_favorable_hot_block_beats_lru():
    # C=4 (sigma=mu=2). Sequence hand-computed: the final reuse of `a` has
    # distinct reuse distance 5 (> C) so LRU misses, but `a` is hot with
    # hot_in=1 (<= mu) so S3-FIFO holds it. See ADAPTIVE_MAIN_GAP_PLAN.md.
    p = _MainGapPredictor(history_factor=10.0)
    p.set_pool_capacity(4)
    feed(p, [b"a", b"b", b"a", b"b", b"c", b"d", b"e", b"f", b"a"])
    assert (p._win_reuse, p._win_lru, p._win_s3) == (3, 2, 3)
    p_lru, p_s3 = p.decide()
    assert p_s3 > p_lru
    assert p_lru == pytest.approx(2 / 3)
    assert p_s3 == pytest.approx(1.0)


def test_predictor_lru_favorable_when_hot_set_exceeds_mu():
    # C=4 (sigma=mu=2). A tight 4-block round-robin: every reuse has dist=3
    # (<= C, LRU hits) but the final hot reuse sees hot_in=3 (> mu), so S3-FIFO
    # would evict it -> LRU wins. Hand-computed in the plan.
    p = _MainGapPredictor(history_factor=10.0)
    p.set_pool_capacity(4)
    feed(p, [b"a", b"b", b"c", b"d", b"a", b"b", b"c", b"d", b"a"])
    assert (p._win_reuse, p._win_lru, p._win_s3) == (5, 5, 3)
    p_lru, p_s3 = p.decide()
    assert p_lru > p_s3
    assert p_lru == pytest.approx(1.0)
    assert p_s3 == pytest.approx(0.6)


def test_predictor_bounds_memory_and_survives_compaction():
    # history_factor=1, C=5 -> history_capacity=5, pos_capacity=10.
    p = _MainGapPredictor(history_factor=1.0)
    p.set_pool_capacity(5)
    feed(p, [bytes([i]) for i in range(20)])  # 20 distinct, far past the cap
    assert len(p._pos) == 5  # tracked-hash count is capped
    assert p._t <= p._pos_capacity  # positions stay dense after compaction
    assert len(p._bit) == p._pos_capacity + 1
    assert len(p._bit_hot) == p._pos_capacity + 1
    # The most-recently-seen hash is still tracked: re-referencing it scores a
    # finite reuse (dist=0), proving compaction preserved ordering.
    p.record_ref(bh(bytes([19])))
    assert (p._win_reuse, p._win_lru, p._win_s3) == (1, 1, 1)


def test_predictor_finalize_capacity_uses_samples():
    p = _MainGapPredictor()
    p.set_pool_capacity(100)  # C=100 until finalize
    for _ in range(4):
        p.sample_capacity(8)
    p.finalize_capacity()
    assert p._C == pytest.approx(8.0)
    assert p._sigma == pytest.approx(4.0)
    assert p._mu == pytest.approx(4.0)
    # No samples next window -> keep the prior.
    p.finalize_capacity()
    assert p._C == pytest.approx(8.0)


def test_predictor_reset_clears_state():
    p = _MainGapPredictor(history_factor=10.0)
    p.set_pool_capacity(4)
    feed(p, [b"a", b"b", b"a"])
    assert p._win_reuse > 0
    p.reset()
    assert p._win_reuse == 0
    assert len(p._pos) == 0
    assert p.decide() == (0.0, 0.0)


def test_predictor_seeds_capacity_from_first_sample():
    # Until the first sample, C is the bootstrap pool size; the first
    # sample_capacity call immediately reseeds C to the live free-block count,
    # so the very first window classifies at effective capacity (no warmup).
    p = _MainGapPredictor()
    p.set_pool_capacity(100)
    assert p._C == 100.0  # bootstrap
    p.sample_capacity(8)  # first live sample
    assert p._C == pytest.approx(8.0)
    assert p._sigma == pytest.approx(4.0)
    assert p._mu == pytest.approx(4.0)


def test_predictor_history_max_caps_tracking_budget():
    # Without a cap, H = round(history_factor * capacity).
    uncapped = _MainGapPredictor(history_factor=2.0)
    uncapped.set_pool_capacity(1000)
    assert uncapped._history_capacity == 2000
    # With a cap, H is clamped (for very large pools).
    capped = _MainGapPredictor(history_factor=2.0, history_max=500)
    capped.set_pool_capacity(1000)
    assert capped._history_capacity == 500
    assert capped._pos_capacity == 1000
    assert len(capped._bit) == 1001


# --------------------------------------------------------------------------- #
# Adaptive main-gap policy (self-tuning LRU <-> S3-FIFO)
# --------------------------------------------------------------------------- #
def make_adaptive(capacity: int = 4, **extra) -> AdaptiveMainGapGPUCachePolicy:
    policy = make_gpu_eviction_policy("adaptive_main_gap", extra)
    assert isinstance(policy, AdaptiveMainGapGPUCachePolicy)
    policy.set_pool_capacity(capacity)
    return policy


def adaptive_free_ids(policy: AdaptiveMainGapGPUCachePolicy) -> set[int]:
    return {b.block_id for b in policy.get_all_free_blocks()}


def force_decision_due(policy: AdaptiveMainGapGPUCachePolicy) -> None:
    # Make the next insert_n treat the decision interval as elapsed, without
    # sleeping (the interval clock is time.monotonic() on the engine thread).
    policy._last_decision_time -= policy._interval_s + 1.0


def test_factory_builds_adaptive_main_gap():
    policy = make_gpu_eviction_policy("adaptive_main_gap")
    assert isinstance(policy, AdaptiveMainGapGPUCachePolicy)
    assert policy.active_policy == "lru"
    # Case-insensitive.
    assert isinstance(
        make_gpu_eviction_policy("ADAPTIVE_MAIN_GAP"),
        AdaptiveMainGapGPUCachePolicy,
    )


def test_factory_reads_extra_config_for_adaptive():
    policy = make_gpu_eviction_policy(
        "adaptive_main_gap",
        {
            "start_policy": "s3_fifo",
            "margin": 0.05,
            "interval_min": 5.0,
            "min_references": 7,
            "history_factor": 3.0,
            "history_max": 4096,
            "consecutive": 2,
            "promote_threshold": 2,
            "freq_cap": 5,
        },
    )
    assert isinstance(policy, AdaptiveMainGapGPUCachePolicy)
    assert policy.active_policy == "s3_fifo"
    assert policy._margin == 0.05
    assert policy._interval_s == 300.0  # 5 minutes in seconds
    assert policy._min_references == 7
    assert policy._consecutive == 2
    assert policy._predictor._history_factor == 3.0
    assert policy._predictor._history_max == 4096
    # S3 knobs forwarded to the inner S3 sub-policy.
    assert policy._s3._promote_threshold == 2
    assert policy._s3._freq_cap == 5


def test_factory_rejects_bad_adaptive_params():
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"start_policy": "lfu"})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"nope": 1})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"history_factor": 0.5})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"interval_min": 0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"min_references": 0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"history_max": 0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"margin": -1.0})
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("adaptive_main_gap", {"consecutive": 0})
    # Inner S3 validation still applies.
    with pytest.raises(ValueError):
        make_gpu_eviction_policy(
            "adaptive_main_gap", {"promote_threshold": 5, "freq_cap": 3}
        )


def test_adaptive_hooks_feed_predictor_and_track_universal_stats():
    policy = make_adaptive()  # LRU active
    block = make_block(0, b"x")
    policy.on_cache(block, 0)  # a miss-fill is a reference
    assert policy._predictor._win_reuse == 0  # first ref, no reuse yet
    assert len(policy._predictor._pos) == 1  # content is now tracked
    policy.touch(block)  # a hit is a reference (a reuse of x)
    assert policy._predictor._win_reuse == 1
    assert block.dyn_freq == 1  # universal stat still tracked by the wrapper
    # Hashless blocks are not content references.
    hashless = make_block(1)
    policy.touch(hashless)
    assert len(policy._predictor._pos) == 1  # unchanged


def _drive_s3_window(
    policy: AdaptiveMainGapGPUCachePolicy,
    hot: KVCacheBlock,
    cold_start: int,
    cycles: int = 4,
    cold_per_cycle: int = 6,
) -> int:
    # Each cycle floods `cold_per_cycle` fresh one-shot cold blocks (> C, so
    # LRU misses the hot reuse) then touches the single hot block (hot_in=0 <=
    # mu, so S3-FIFO holds it). Strongly S3-favorable, robust across windows.
    cid = cold_start
    for _ in range(cycles):
        for _ in range(cold_per_cycle):
            policy.touch(make_block(cid, str(cid).encode()))
            cid += 1
        policy.touch(hot)
    return cid


def test_adaptive_no_switch_before_interval_elapses():
    policy = make_adaptive(min_references=1)  # default 15-min interval
    hot = make_block(50, b"H")
    _drive_s3_window(policy, hot, cold_start=1000)
    policy.insert_n([])  # interval has not elapsed -> no decision
    assert policy.active_policy == "lru"


def test_adaptive_switches_to_s3_and_preserves_free_set():
    policy = make_adaptive(min_references=1, consecutive=1, history_factor=4.0)
    # A resident free set that must survive the switch untouched.
    free_set = [make_block(300 + i, str(300 + i).encode()) for i in range(4)]
    policy.insert_n(free_set)
    before = adaptive_free_ids(policy)
    hot = make_block(50, b"H")
    _drive_s3_window(policy, hot, cold_start=1000)
    assert policy.active_policy == "lru"
    force_decision_due(policy)
    policy.insert_n([])  # interval elapsed -> decision fires -> switch
    assert policy.active_policy == "s3_fifo"
    assert adaptive_free_ids(policy) == before  # no block lost across the swap


def test_adaptive_consecutive_hysteresis_delays_switch():
    policy = make_adaptive(min_references=1, consecutive=2, history_factor=4.0)
    free_set = [make_block(300 + i, str(300 + i).encode()) for i in range(4)]
    policy.insert_n(free_set)
    hot = make_block(50, b"H")
    cid = _drive_s3_window(policy, hot, cold_start=1000)
    force_decision_due(policy)
    policy.insert_n([])  # window 1: challenger wins once, not enough
    assert policy.active_policy == "lru"
    assert policy._wins == 1
    _drive_s3_window(policy, hot, cold_start=cid)
    force_decision_due(policy)
    policy.insert_n([])  # window 2: second win -> switch
    assert policy.active_policy == "s3_fifo"


def test_adaptive_min_references_guards_switch():
    # A window that "wins" but sees fewer than min_references reuses must not
    # switch (too little signal), and resets the challenger's win streak.
    policy = make_adaptive(min_references=1000, consecutive=1, history_factor=4.0)
    hot = make_block(50, b"H")
    _drive_s3_window(policy, hot, cold_start=1000)  # only a few reuses
    force_decision_due(policy)
    policy.insert_n([])
    assert policy.active_policy == "lru"
    assert policy._wins == 0


def test_adaptive_reset_clears_predictor_and_counters():
    policy = make_adaptive()
    block = make_block(0, b"x")
    policy.on_cache(block, 0)
    policy.touch(block)
    assert policy._predictor._win_reuse > 0
    policy.reset()
    assert policy._wins == 0
    assert policy._predictor._win_reuse == 0
    assert len(policy._predictor._pos) == 0


def test_adaptive_block_pool_end_to_end_builds_and_runs():
    pool = BlockPool(
        num_gpu_blocks=64,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="adaptive_main_gap",
        eviction_policy_extra_config={"interval_min": 5.0},
    )
    assert isinstance(pool._policy, AdaptiveMainGapGPUCachePolicy)
    assert pool.get_active_eviction_policy() == "lru"


# --------------------------------------------------------------------------- #
# ARC (Adaptive Replacement Cache)
# --------------------------------------------------------------------------- #
def make_arc(capacity: int = 8) -> ARCGPUCachePolicy:
    policy = make_gpu_eviction_policy("arc")
    assert isinstance(policy, ARCGPUCachePolicy)
    policy.set_pool_capacity(capacity)
    return policy


def arc_admit(policy: ARCGPUCachePolicy, block: KVCacheBlock) -> None:
    """Cache fresh content, then free the block (ends free in T1 or T2)."""
    policy.on_cache(block, 0)
    policy.insert(block)


def arc_hit(policy: ARCGPUCachePolicy, block: KVCacheBlock) -> None:
    """A cache hit on a free block, then free it — promotes it to T2 (Case I)."""
    policy.remove(block)  # BlockPool removes before touch
    policy.touch(block)
    policy.insert(block)


def test_factory_builds_arc():
    policy = make_gpu_eviction_policy("arc")
    assert isinstance(policy, ARCGPUCachePolicy)
    assert isinstance(make_gpu_eviction_policy("ARC"), ARCGPUCachePolicy)


def test_factory_arc_rejects_options():
    with pytest.raises(ValueError):
        make_gpu_eviction_policy("arc", {"p": 1})


def test_arc_fresh_miss_enters_t1():
    policy = make_arc()
    block = make_block(1, b"a")
    policy.on_cache(block, 0)  # Case IV
    assert policy.n_t1 == 1 and policy.n_t2 == 0
    policy.insert(block)
    assert [b.block_id for b in policy._t1_free.get_all_free_blocks()] == [1]


def test_arc_hit_promotes_to_t2():
    policy = make_arc()
    block = make_block(1, b"a")
    arc_admit(policy, block)  # T1
    arc_hit(policy, block)  # Case I -> T2
    assert policy.n_t1 == 0 and policy.n_t2 == 1
    assert [b.block_id for b in policy._t2_free.get_all_free_blocks()] == [1]


def test_arc_replace_evicts_t1_when_over_target():
    # a re-hit -> T2; b,c,d fresh -> T1. n_t1=3 > p=0 -> evict T1 LRU (=b), → B1.
    policy = make_arc()
    a = make_block(1, b"a")
    arc_admit(policy, a)
    arc_hit(policy, a)  # a -> T2
    for bid, content in [(2, b"b"), (3, b"c"), (4, b"d")]:
        arc_admit(policy, make_block(bid, content))
    assert (policy.n_t1, policy.n_t2, policy.p) == (3, 1, 0.0)
    victims = policy.evict_n(1)
    assert [v.block_id for v in victims] == [2]  # b, the T1 LRU
    assert policy.b1_size == 1 and policy.b2_size == 0
    assert policy.n_t1 == 2


def test_arc_b1_ghost_hit_grows_p_and_promotes_to_t2():
    policy = make_arc()
    b = make_block(1, b"b")
    arc_admit(policy, b)  # T1
    policy.evict_n(1)  # evict b -> B1 (hash of b"b")
    assert policy.b1_size == 1 and policy.n_t1 == 0
    # Re-cache the same content into a new block: a B1 ghost hit (Case II).
    b_again = make_block(2, b"b")
    policy.on_cache(b_again, 0)
    assert policy.p == 1.0  # delta = max(1, |B2|/|B1|) = 1
    assert policy.b1_size == 0
    assert policy.n_t2 == 1 and policy.n_t1 == 0  # promoted to T2


def test_arc_b2_ghost_hit_shrinks_p():
    policy = make_arc()
    # Lift p above 0 with a B1 ghost hit.
    x = make_block(1, b"x")
    arc_admit(policy, x)  # x free in T1
    policy.evict_n(1)  # evict x -> B1
    x2 = make_block(2, b"x")  # same content
    policy.on_cache(x2, 0)  # B1 ghost hit -> p = 1, x2 labelled T2
    policy.insert(x2)  # free x2 into T2
    assert policy.p == 1.0
    # Create a B2 entry, then re-reference it as a ghost hit (Case III).
    policy.evict_n(1)  # n_t1=0 <= p=1 -> evict T2 (x2) -> B2
    assert policy.b2_size == 1
    x3 = make_block(3, b"x")
    policy.on_cache(x3, 0)  # B2 ghost hit -> p shrinks
    assert policy.p < 1.0


def test_arc_replace_fallback_to_t2_when_t1_pinned():
    # a is in T1 but PINNED (never freed) so _t1_free is empty; b is free in T2.
    # n_t1=1 > p=0 prefers T1, but with no free T1 block it must evict T2.
    policy = make_arc()
    a = make_block(1, b"a")
    policy.on_cache(a, 0)  # a -> T1, pinned (not inserted)
    b = make_block(2, b"b")
    arc_admit(policy, b)
    arc_hit(policy, b)  # b -> T2 (free)
    assert policy.n_t1 == 1 and policy.n_t2 == 1
    assert policy._t1_free.num_free_blocks == 0
    victims = policy.evict_n(1)
    assert [v.block_id for v in victims] == [2]  # fell back to T2
    assert policy.b2_size == 1 and policy.b1_size == 0  # ghost follows real list
    assert policy.n_t2 == 0 and policy.n_t1 == 1  # pinned a still counted
    assert policy.p == 0.0  # eviction never moves p


def test_arc_hashless_evicted_before_cached():
    policy = make_arc()
    arc_admit(policy, make_block(1, b"a"))  # cached T1
    policy.insert(make_block(2))  # hashless
    victims = policy.evict_n(1)
    assert [v.block_id for v in victims] == [2]  # hashless first
    assert policy.n_t1 == 1  # cached block untouched


def test_arc_remove_absent_block_is_noop():
    policy = make_arc()
    present = make_block(1, b"a")
    arc_admit(policy, present)
    policy.remove(make_block(2, b"b"))  # never inserted; must not raise
    assert len(policy) == 1


def test_arc_reset_clears_ghosts_labels_and_counters():
    policy = make_arc()
    arc_admit(policy, make_block(1, b"a"))
    arc_admit(policy, make_block(2, b"b"))
    policy.evict_n(1)  # -> B1
    assert policy.b1_size == 1
    # Prefix-cache reset: BlockPool clears hashes first, then calls reset().
    for blk in policy.get_all_free_blocks():
        blk.reset_hash()
    policy.reset()
    assert policy.b1_size == 0 and policy.b2_size == 0
    assert policy.n_t1 == 0 and policy.n_t2 == 0 and policy.p == 0.0


def test_arc_evict_more_than_available_raises():
    policy = make_arc()
    arc_admit(policy, make_block(1, b"a"))
    with pytest.raises(ValueError):
        policy.evict_n(2)


def test_arc_block_pool_end_to_end_builds_and_allocates():
    pool = BlockPool(
        num_gpu_blocks=16,
        enable_caching=True,
        hash_block_size=16,
        eviction_policy="arc",
    )
    assert isinstance(pool._policy, ARCGPUCachePolicy)
    blocks = pool.get_new_blocks(4)
    assert len(blocks) == 4


def test_adaptive_switch_to_s3_preserves_ghost_and_seeds_freq():
    # Regression: the adaptive policy's switch to S3 must NOT wipe S3's
    # accumulated protection. Its _rebuild_s3 override preserves the ghost and
    # seeds each main block's second-chance budget from lifetime hotness, so a
    # switch does not strand the rebuilt main queue at freq 0 (evicted on the
    # first pass) -- the cause of the collapse under policy oscillation. The base
    # DynamicGPUCachePolicy keeps its original destructive rebuild (untouched).
    policy = make_gpu_eviction_policy(
        "adaptive_main_gap", {"promote_threshold": 1, "freq_cap": 3}
    )
    assert isinstance(policy, AdaptiveMainGapGPUCachePolicy)
    policy.set_pool_capacity(100)
    hot = make_block(1, b"hot")
    policy.insert(hot)
    policy.remove(hot)
    policy.touch(hot)
    policy.insert(hot)  # dyn_freq 1
    policy.remove(hot)
    policy.touch(hot)
    policy.insert(hot)  # dyn_freq 2 -> main on rebuild
    # A fingerprint in S3's ghost (as if content was demoted from small earlier).
    ghost_hash = make_block_hash_with_group_id(BlockHash(b"ghosted"), 0)
    policy._s3._ghost[ghost_hash] = None

    assert policy.set_active_policy("s3_fifo") is True
    assert hot.s3_queue == _S3_MAIN
    assert hot.s3_freq == 2  # min(dyn_freq=2, freq_cap=3), not 0
    assert ghost_hash in policy._s3._ghost  # ghost survives the switch


def test_dynamic_switch_to_s3_stays_destructive():
    # The base dynamic policy is deliberately left with the original rebuild:
    # ghost cleared and s3_freq zeroed on every switch. This pins that contrast
    # with the adaptive override above.
    policy = make_dynamic(promote_threshold=1, freq_cap=3)
    hot = make_block(1, b"hot")
    policy.insert(hot)
    policy.remove(hot)
    policy.touch(hot)
    policy.insert(hot)  # dyn_freq 1 -> main on rebuild
    ghost_hash = make_block_hash_with_group_id(BlockHash(b"ghosted"), 0)
    policy._s3._ghost[ghost_hash] = None

    assert policy.set_active_policy("s3_fifo") is True
    assert hot.s3_queue == _S3_MAIN
    assert hot.s3_freq == 0  # base zeroes it
    assert ghost_hash not in policy._s3._ghost  # base clears the ghost


