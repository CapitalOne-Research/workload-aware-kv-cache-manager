# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end equivalence of GPU eviction policies.

When the KV cache never fills, nothing is evicted, so LRU and LFU allocate and
free the *same* blocks and hit the *same* prefix-cache entries -- the policy
only controls the *order* of free candidates, which never affects the
computation unless an eviction actually happens. Greedy output must therefore
be token-for-token identical across policies.

Run on a SINGLE GPU (TP=1, the VllmRunner default). Tensor parallelism makes the
per-layer all-reduce non-bitwise-reproducible, which flips near-tied greedy
tokens run to run and would mask this comparison; TP=1 avoids it. To validate
with a local model, point at it (defaults to a tiny Gemma):

    VLLM_EVICTION_TEST_MODEL=/path/to/gemma \
      pytest tests/v1/core/test_eviction_policy_e2e.py -v
"""

import os

import pytest

from ...models.utils import check_outputs_equal

MODEL = os.environ.get(
    "VLLM_EVICTION_TEST_MODEL", "hmellor/tiny-random-Gemma2ForCausalLM"
)

# A long shared prefix guarantees prefix-cache hits: later prompts reuse the
# blocks the earlier ones cached, exercising the policy's insert/touch paths.
_PREFIX = "The quick brown fox jumps over the lazy dog. " * 20
PROMPTS = [
    _PREFIX + "Question: what animal jumps? Answer:",
    _PREFIX + "Question: what is lazy? Answer:",
    _PREFIX + "Question: summarize in one word. Answer:",
    _PREFIX + "Question: is the dog awake? Answer:",
]


def _generate(vllm_runner, policy: str, max_tokens: int):
    with vllm_runner(
        MODEL,
        enable_prefix_caching=True,
        gpu_eviction_policy=policy,
        enforce_eager=True,
    ) as llm:
        return llm.generate_greedy(PROMPTS, max_tokens)


@pytest.mark.parametrize("max_tokens", [32])
def test_lru_lfu_match_when_cache_not_full(vllm_runner, max_tokens):
    """LFU must match LRU exactly when no eviction happens."""
    lru = _generate(vllm_runner, "lru", max_tokens)
    lfu = _generate(vllm_runner, "lfu", max_tokens)
    check_outputs_equal(
        outputs_0_lst=lru,
        outputs_1_lst=lfu,
        name_0="lru",
        name_1="lfu",
    )


@pytest.mark.parametrize("max_tokens", [32])
def test_lru_is_deterministic_across_runs(vllm_runner, max_tokens):
    """Sanity/noise-floor: two identical LRU runs must match at TP=1. If this
    fails, the environment is nondeterministic (e.g. TP>1) and the policy
    comparison above is meaningless -- switch to the num_cached_tokens signal."""
    first = _generate(vllm_runner, "lru", max_tokens)
    second = _generate(vllm_runner, "lru", max_tokens)
    check_outputs_equal(
        outputs_0_lst=first,
        outputs_1_lst=second,
        name_0="lru_run1",
        name_1="lru_run2",
    )
