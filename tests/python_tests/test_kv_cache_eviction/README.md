# KV Cache Eviction Tests

Tests for KV cache eviction strategies in the OpenVINO GenAI continuous batching pipeline.

## Test Files

| File | Description |
|------|-------------|
| `test_kv_cache_eviction_1.py` | Core eviction tests: similarity between optimized/unoptimized generation, dynamic memory allocation, and LongBench quality evaluation. Covers both `NORM_SUM` and `ADAPTIVE_RKV` aggregation modes. |
| `test_kv_cache_eviction_2.py` | Head-to-head comparisons: KVCrush vs SnapKV baseline, and Adaptive RKV vs SnapKV baseline on LongBench subsets. |
| `test_three_way_comparison.py` | Three-way quality comparison: no eviction (baseline) vs `NORM_SUM` vs `ADAPTIVE_RKV`. Prints a table showing score drop and cache optimization ratio for each method. |
| `kv_cache_eviction_utils.py` | Shared utilities (e.g. `get_scheduler_config`). |

## KV Cache Layout

All eviction strategies operate on the same cache structure. The KV cache for each sequence is logically divided into three contiguous areas:

```
|<-- start_size -->|<-- evictable area -->|<-- recent_size -->|
|   sink tokens    |  candidates for      |  most recent      |
|   (always kept)  |  eviction            |  tokens (kept)    |
```

- **start_size** — Initial "sink" tokens that are always retained (capture global context).
- **recent_size** — Most recently generated tokens, always retained (capture local context).
- **evictable area** = `max_cache_size - start_size - recent_size` — Tokens in this area are scored and the least important ones are evicted when the cache is full.

When the total number of cached tokens exceeds `max_cache_size`, the eviction algorithm selects which tokens in the evictable area to remove. The `aggregation_mode` determines how token importance scores are computed.

## Eviction Modes Tested

### 1. NORM_SUM (SnapKV-style baseline)

```
Score aggregation:  score[token] += attention_weight / token_lifetime
Eviction decision:  evict tokens with lowest accumulated score
```

Each token's importance is the sum of its attention scores across all generation steps, normalized by how long the token has been in the cache (its "lifetime"). This prevents older tokens from accumulating unfairly high scores simply by being present longer. Tokens with the lowest normalized score are evicted first.

**Parameters used in tests:**
- `start_size=32, recent_size=32, max_cache_size=96` (short config, aggressive eviction)
- `start_size=32, recent_size=128, max_cache_size=672` (LongBench config)
- `snapkv_window_size=8` (default, number of recent steps to aggregate scores from)

### 2. SUM

```
Score aggregation:  score[token] += attention_weight
Eviction decision:  evict tokens with lowest accumulated score
```

Same as NORM_SUM but without the lifetime normalization. Older tokens that consistently receive attention accumulate higher scores over time, which can bias retention toward earlier tokens. Defined in the enum but not directly tested in these test files.

### 3. ADAPTIVE_RKV

```
Score aggregation:  attention-mass-based scoring within a sliding window
Diversity scoring:  cosine similarity between key vectors in evictable area
Eviction decision:  combine attention importance + key diversity to decide eviction
```

A two-phase approach:

**Phase A — Attention mass preservation (genai side, `cache_eviction.cpp`):**
Attention scores are aggregated using a max-pool within a configurable window of recent generation steps. The algorithm identifies which tokens capture a target fraction (`attention_mass`, e.g. 0.9) of the total attention mass and marks them as important.

**Phase B — Key-vector diversity scoring (OpenVINO CPU plugin, `executor_pa.cpp::compute_adaptive_rkv_diversity`):**
For each sequence, the kernel gathers key vectors from the evictable area, L2-normalizes them, computes pairwise cosine similarities, and produces a per-block diversity score. Tokens with redundant (highly similar) key representations are preferred for eviction, while tokens with diverse, unique representations are retained.

The final eviction decision combines both signals: tokens that are both low-importance (by attention mass) and low-diversity (redundant keys) are evicted first. This aims to preserve output quality better than attention-only methods under the same cache budget.

**Parameters used in tests:**
- `start_size=32, recent_size=32, max_cache_size=96` (short config)
- `start_size=32, recent_size=128, max_cache_size=672` (LongBench config)
- `attention_mass=0.9` (retain tokens covering 90% of attention mass)
- `window_size=8` (aggregate scores over last 8 generation steps)
- `snapkv_window_size=0` (disable SnapKV-style aggregation, use adaptive method instead)

### 4. KVCrush

```
Scoring:            NORM_SUM attention score aggregation (same as mode 1)
Compression:        locality-sensitive hashing to cluster similar KV pairs
Eviction decision:  merge clusters, keep representative tokens
```

KVCrush extends the NORM_SUM baseline by adding a compression step using locality-sensitive hashing (LSH). Key vectors are hashed relative to configurable "anchor points" to identify clusters of similar tokens. Within each cluster, representative tokens are kept and redundant ones are merged/evicted.

**Parameters used in tests (`test_kv_cache_eviction_2.py`):**
- `budget` — Number of blocks to retain via KVCrush (e.g. 2 or 8, per-subset tuned)
- `anchor_point_mode` — How LSH anchor points are formed (ALTERNATING in tests)
- Baseline comparison uses `KVCrushConfig(budget=0)` (KVCrush disabled, pure SnapKV)

### 5. Cache Rotation (orthogonal option)

Not a standalone eviction mode but an additional optimization tested via the `apply_rotation` parameter in `test_kv_cache_eviction_1.py`. When enabled, key vectors in evicted blocks are "rotated" (transformed) before removal to preserve positional encoding information for remaining tokens. Can be combined with any aggregation mode.

## LongBench Subsets

These tests use subsets from the [LongBench](https://github.com/THUDM/LongBench) benchmark. Each subset tests a different long-context capability:

| Subset | Task | Description | Metric | Max New Tokens |
|--------|------|-------------|--------|----------------|
| `samsum` | Dialogue summarization | Given a multi-turn conversation, produce a concise summary. (SAMSum corpus) | ROUGE-L | 128 |
| `trec` | Question classification | Given few-shot examples, classify a question into categories (person, location, number, etc.). (TREC dataset) | Classification accuracy | 64 |
| `qasper` | Scientific paper QA | Given a full scientific paper and a question, answer concisely or respond "unanswerable". | Token-level F1 | 128 |
| `hotpotqa` | Multi-hop QA | Given multiple Wikipedia passages, answer a question that requires reasoning across passages. | Token-level F1 | 32 |

## Running the Tests

```bash
cd /path/to/openvino.genai/tests/python_tests

# Run all eviction tests
pytest test_kv_cache_eviction/ -v

# Run a specific test file
pytest test_kv_cache_eviction/test_three_way_comparison.py -v

# Run a single subset
pytest test_kv_cache_eviction/test_three_way_comparison.py::test_three_way_quality_comparison[samsum] -v
```

## Using a Custom Model

By default, tests download `HuggingFaceTB/SmolLM2-135M-Instruct` (135M params, fast but limited quality differentiation). For realistic evaluation with a larger model, set the `OV_MODEL_PATH` environment variable to a local pre-converted OpenVINO model directory:

```bash
export OV_MODEL_PATH=/path/to/models/deepseek-r1-distill-qwen-7b/pytorch/ov/FP16/
pytest test_kv_cache_eviction/test_three_way_comparison.py -v
```

Recommended models for meaningful eviction testing (7B+ parameters, good long-context behavior):
- `deepseek-r1-distill-qwen-7b` — 7B params, Qwen2.5 architecture, distilled from DeepSeek-R1
- `Qwen/Qwen2.5-7B-Instruct` — 7B params, strong long-context performance
- `meta-llama/Llama-3.1-8B-Instruct` — 8B params, 128K context window

Models should be exported to OpenVINO IR format (FP16) before use. See the [OpenVINO GenAI documentation](https://github.com/openvinotoolkit/openvino.genai) for export instructions.

## Prerequisites

- `openvino_genai` package installed or on `PYTHONPATH`
- Python packages: `datasets`, `tqdm`, `rouge`, `pytest`
- For `test_kv_cache_eviction_1.py`: additionally `whowhatbench`
- Network access to HuggingFace (for model/dataset downloads, unless using local model via `OV_MODEL_PATH`)
