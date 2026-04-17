# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Modular eviction comparison test.
#
# Each eviction strategy is registered as an EvictionMethod. To add a new method,
# define a new EvictionMethod instance and append it to EVICTION_METHODS — no other
# code changes are needed.

import gc
import os
import sys
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datasets
from tqdm import tqdm

from openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationConfig,
    CacheEvictionConfig,
    AggregationMode,
    KVCrushConfig,
    KVCrushAnchorPointMode,
)

from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model
from utils.longbench import dataset2maxlen, evaluate, preprocess_prompt, post_process_pred
from kv_cache_eviction_utils import get_scheduler_config


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Set OV_MODEL_PATH to a local pre-converted OpenVINO model for realistic eval:
#   export OV_MODEL_PATH=/path/to/deepseek-r1-distill-qwen-7b/pytorch/ov/FP16/
REALISTIC_MODEL_PATH = os.environ.get("OV_MODEL_PATH")


def _get_model_path_and_name():
    """Return (models_path, model_name) using local model if available, else download default."""
    if REALISTIC_MODEL_PATH and Path(REALISTIC_MODEL_PATH).exists():
        models_path = Path(REALISTIC_MODEL_PATH)
        model_name = models_path.name
    else:
        models_path = download_and_convert_model(DEFAULT_MODEL_ID).models_path
        model_name = "/".join(models_path.parts[-2:])
    print(f"Using model: {model_name} at {models_path}")
    return models_path, model_name


# ---------------------------------------------------------------------------
# Eviction method registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvictionMethod:
    """A pluggable eviction strategy definition.

    Attributes:
        name:          Short identifier used in test IDs and output tables.
        description:   One-line description printed in reports.
        build_config:  Callable(start_size, recent_size, max_cache_size) -> CacheEvictionConfig.
                       Return None for the no-eviction baseline.
    """
    name: str
    description: str
    build_config: Callable[[int, int, int], CacheEvictionConfig | None]


def _build_norm_sum(start_size, recent_size, max_cache_size):
    """NORM_SUM: attention scores normalized by token lifetime."""
    return CacheEvictionConfig(
        start_size=start_size,
        recent_size=recent_size,
        max_cache_size=max_cache_size,
        aggregation_mode=AggregationMode.NORM_SUM,
    )


def _build_adaptive_rkv(start_size, recent_size, max_cache_size):
    """ADAPTIVE_RKV: attention mass preservation + key-vector diversity."""
    config = CacheEvictionConfig(
        start_size=start_size,
        recent_size=recent_size,
        max_cache_size=max_cache_size,
        aggregation_mode=AggregationMode.ADAPTIVE_RKV,
        snapkv_window_size=0,
    )
    config.adaptive_rkv_config.attention_mass = 0.9
    config.adaptive_rkv_config.window_size = 8
    return config


def _build_kvcrush(start_size, recent_size, max_cache_size):
    """KVCrush: NORM_SUM + locality-sensitive hashing compression."""
    return CacheEvictionConfig(
        start_size=start_size,
        recent_size=recent_size,
        max_cache_size=max_cache_size,
        aggregation_mode=AggregationMode.NORM_SUM,
        snapkv_window_size=8,
        kvcrush_config=KVCrushConfig(budget=2, anchor_point_mode=KVCrushAnchorPointMode.ALTERNATING),
    )


# ---- Registry ----
# To add a new eviction method, append an EvictionMethod here.
# The test will automatically include it in the comparison.

EVICTION_METHODS = [
    EvictionMethod(
        name="no_eviction",
        description="Baseline — no cache eviction",
        build_config=lambda _start, _recent, _max: None,
    ),
    EvictionMethod(
        name="norm_sum",
        description="SnapKV-style normalized attention score accumulation",
        build_config=_build_norm_sum,
    ),
    EvictionMethod(
        name="adaptive_rkv",
        description="Attention mass preservation + key-vector diversity scoring",
        build_config=_build_adaptive_rkv,
    ),
    EvictionMethod(
        name="kvcrush",
        description="NORM_SUM + LSH-based KV pair clustering/compression",
        build_config=_build_kvcrush,
    ),
]



# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """LongBench subset configuration.

    Attributes:
        subset:         LongBench subset name.
        num_samples:    Number of samples to evaluate.
        start_size:     Sink token count (always retained).
        recent_size:    Recent token count (always retained).
        max_cache_size: Total cache budget — eviction triggers when exceeded.
    """
    subset: str
    num_samples: int = 16
    start_size: int = 32
    recent_size: int = 128
    max_cache_size: int = 672


# LongBench subsets:
#   samsum   — dialogue summarization (SAMSum). Metric: ROUGE-L.
#   trec     — question type classification (TREC). Metric: classification accuracy.
#   qasper   — scientific paper QA. Metric: token-level F1.
#   hotpotqa — multi-hop QA across Wikipedia passages. Metric: token-level F1.
DATASET_CONFIGS = [
    DatasetConfig("samsum"),
    DatasetConfig("trec"),
    DatasetConfig("qasper"),
    DatasetConfig("hotpotqa"),
]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(pipeline, prompts, generation_config, subset, model_name, seqs_per_request=16):
    """Run a pipeline on all prompts and return post-processed predictions."""
    answers = [None] * len(prompts)
    batch = []
    for p_idx, prompt in enumerate(prompts):
        batch.append(prompt)

        if len(batch) == seqs_per_request or p_idx == len(prompts) - 1:
            outputs = pipeline.generate(batch, [generation_config] * len(batch))
            for i, output in enumerate(outputs, start=p_idx - len(batch) + 1):
                answers[i] = post_process_pred(output.m_generation_ids[0], subset, model_name)
            batch.clear()
    return answers


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    score: float
    max_cache: float
    avg_cache: float


def print_comparison_table(subset: str, results: dict[str, MethodResult]):
    """Print a formatted comparison table for all evaluated methods.

    Columns:
        Score    — LongBench task score (metric depends on subset: ROUGE-L for samsum,
                   classification accuracy for trec, token-level F1 for qasper/hotpotqa).
                   Higher is better. The no_eviction row is the quality ceiling.
        Drop     — Quality drop vs no_eviction baseline (= baseline_score - score).
                   +0.00 means no degradation; larger positive = more quality lost.
        MaxCache — Peak KV cache usage as a percentage of total allocated cache (0-100).
                   Lower means the method is more memory-efficient at peak load.
        AvgCache — Average KV cache usage over the entire run (0-100).
                   Lower means the method sustained less memory pressure overall.
    """
    baseline_score = results["no_eviction"].score

    print("\n" + "=" * 90)
    print(f"  EVICTION METHOD COMPARISON — {subset}")
    print("=" * 90)
    print(f"  Score:    task quality (higher=better)    Drop:     quality loss vs baseline (lower=better)")
    print(f"  MaxCache: peak cache usage %              AvgCache: average cache usage %")
    print("-" * 90)
    print(f"{'Method':<20} {'Description':<36} {'Score':>7} {'Drop':>7} {'MaxCache':>9} {'AvgCache':>9}")
    print("-" * 90)

    method_lookup = {m.name: m for m in EVICTION_METHODS}
    for name in results:
        r = results[name]
        drop = baseline_score - r.score
        desc = method_lookup[name].description[:35] if name in method_lookup else ""
        print(f"{name:<20} {desc:<36} {r.score:>7.2f} {drop:>+7.2f} {r.max_cache:>9.3f} {r.avg_cache:>9.3f}")
    print("-" * 90)

    # Summary: rank eviction methods by quality drop (ascending = better)
    eviction_results = {k: v for k, v in results.items() if k != "no_eviction"}
    if eviction_results:
        ranked = sorted(eviction_results.items(), key=lambda kv: baseline_score - kv[1].score)
        print("  Quality ranking (less drop = better):")
        for rank, (name, r) in enumerate(ranked, 1):
            drop = baseline_score - r.score
            cache_ratio = results["no_eviction"].max_cache / max(r.max_cache, 1e-9)
            print(f"    {rank}. {name:<18} drop: {drop:>+7.2f}   cache ratio: {cache_ratio:.2f}x")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform in ("win32", "darwin"),
    reason="doesn't work on win due to optimum-intel export bug, segfault on mac",
)
@pytest.mark.parametrize(
    "dataset_cfg",
    DATASET_CONFIGS,
    ids=[c.subset for c in DATASET_CONFIGS],
)
def test_eviction_comparison(dataset_cfg: DatasetConfig):
    """
    Compare all registered eviction methods against the no-eviction baseline
    on a LongBench subset.

    Each method is evaluated independently. Results are printed as a comparison
    table with score, quality drop, and cache usage.
    """
    device = "CPU"
    num_kv_blocks = 1000
    models_path, model_name = _get_model_path_and_name()

    subset = dataset_cfg.subset
    max_new_tokens = dataset2maxlen[subset]

    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens

    # Load dataset once, reuse for all methods
    data = datasets.load_dataset(
        "zai-org/LongBench", subset,
        split=f"test[:{dataset_cfg.num_samples}]",
        revision="8cbd1",
    )
    prompts = [preprocess_prompt(sample, subset, model_name) for sample in data]
    ground_truth = [
        {"answers": sample["answers"], "all_classes": sample["all_classes"]}
        for sample in data
    ]

    results: dict[str, MethodResult] = {}

    for method in EVICTION_METHODS:
        sched_cfg = get_scheduler_config(num_kv_blocks)
        eviction_config = method.build_config(
            dataset_cfg.start_size, dataset_cfg.recent_size, dataset_cfg.max_cache_size
        )
        if eviction_config is not None:
            sched_cfg.use_cache_eviction = True
            sched_cfg.cache_eviction_config = eviction_config

        pipeline = ContinuousBatchingPipeline(
            models_path, sched_cfg, device, {}, get_default_llm_properties()
        )

        preds = run_pipeline(pipeline, prompts, generation_config, subset, model_name)

        eval_data = [
            {
                "pred": pred,
                "answers": ground_truth[i]["answers"],
                "all_classes": ground_truth[i]["all_classes"],
            }
            for i, pred in enumerate(preds)
        ]
        score = evaluate(eval_data, subset)
        metrics = pipeline.get_metrics()

        results[method.name] = MethodResult(
            score=score,
            max_cache=metrics.max_cache_usage,
            avg_cache=metrics.avg_cache_usage,
        )

        del pipeline
        gc.collect()

    print_comparison_table(subset, results)
