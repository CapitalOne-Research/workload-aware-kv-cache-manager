# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the dynamic-policy shadow-trace tap."""

from vllm.v1.core.sched.shadow_trace import ShadowTraceTap


def test_empty_drain_reports_capacity_as_free():
    tap = ShadowTraceTap(num_gpu_blocks=1000, block_size=16)
    payload = tap.drain()
    assert payload["num_gpu_blocks"] == 1000
    assert payload["block_size"] == 16
    assert payload["num_requests"] == 0
    assert payload["requests"] == []
    # With no samples, free defaults to the full pool.
    assert payload["avg_free_blocks"] == 1000.0


def test_records_requests_and_averages_free_blocks():
    tap = ShadowTraceTap(num_gpu_blocks=1000, block_size=16)
    tap.record_finished(1.5, [b"\x00" * 8, b"\x11" * 8, b"\x22" * 8], 32)
    tap.record_finished(2.5, [b"\xaa" * 8], 100)
    tap.sample_free_blocks(400)
    tap.sample_free_blocks(600)

    payload = tap.drain()
    assert payload["num_requests"] == 2
    assert payload["avg_free_blocks"] == 500.0
    first = payload["requests"][0]
    assert first["arrival_time"] == 1.5
    assert len(first["block_hashes"]) == 3
    assert all(isinstance(h, int) for h in first["block_hashes"])
    # 32 prompt tokens / block_size 16 = 2 prompt blocks (of 3 total).
    assert first["num_prompt_blocks"] == 2
    # 100 // 16 = 6 prompt blocks, capped at the single hash present.
    assert payload["requests"][1]["num_prompt_blocks"] == 1


def test_drain_rolls_the_window():
    tap = ShadowTraceTap(num_gpu_blocks=10, block_size=16)
    tap.record_finished(0.0, [b"a" * 8], 16)
    tap.sample_free_blocks(3)
    tap.drain()
    assert len(tap) == 0
    payload = tap.drain()
    assert payload["num_requests"] == 0
    assert payload["avg_free_blocks"] == 10.0


def test_bounded_buffer_drops_oldest():
    tap = ShadowTraceTap(num_gpu_blocks=10, block_size=16, max_requests=2)
    for i in range(4):
        tap.record_finished(float(i), [bytes([i]) * 8], 16)
    payload = tap.drain()
    assert payload["num_requests"] == 2
    assert [r["arrival_time"] for r in payload["requests"]] == [2.0, 3.0]


def test_same_content_yields_same_opaque_id():
    tap = ShadowTraceTap(num_gpu_blocks=10, block_size=16)
    tap.record_finished(0.0, [b"deadbeef"], 16)
    tap.record_finished(1.0, [b"deadbeef"], 16)
    payload = tap.drain()
    ids0 = payload["requests"][0]["block_hashes"]
    ids1 = payload["requests"][1]["block_hashes"]
    assert ids0 == ids1  # prefix-sharing structure is preserved


def test_zero_block_size_yields_no_prompt_blocks():
    tap = ShadowTraceTap(num_gpu_blocks=10, block_size=0)
    tap.record_finished(0.0, [b"a" * 8, b"b" * 8], 32)
    payload = tap.drain()
    assert payload["requests"][0]["num_prompt_blocks"] == 0
