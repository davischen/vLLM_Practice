"""Chapter 6 vLLM optimization techniques homework.

This local lab implements the chapter's essential LLM optimization ideas with
two layers:

1. Synthetic comparisons that run on any machine.
2. vLLM command builders for running real experiments on a Linux/CUDA machine.

Covered concepts:
    - Dynamic batching versus continuous batching.
    - max-num-seqs and max-num-batched-tokens.
    - Chunked prefill and the TTFT/ITL trade-off.
    - KV-cache size for MHA, GQA, and MQA.
    - PagedAttention-style memory allocation versus contiguous allocation.
    - Quantization memory savings.
    - Prefix caching, prompt formatting, tenant isolation, and cache-aware routing.

Examples:
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section commands
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section experiment-plan
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section batching
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section chunked-prefill
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section attention
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section paged-attention
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section quantization
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section prefix-cache
    python3 -B examples/ch6_vllm_optimization_techniques_hw.py --section all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_QUANT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
GPTQ_MODEL_PLACEHOLDER = "YOUR_GPTQ_MODEL_ID"
FP8_MODEL_PLACEHOLDER = "YOUR_FP8_MODEL_ID"


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Request:
    request_id: int
    arrival_ms: float
    input_tokens: int
    output_tokens: int
    prefix_id: int | None = None


@dataclass
class ServingMetrics:
    strategy: str
    requests: int
    total_input_tokens: int
    total_output_tokens: int
    duration_ms: float
    request_throughput_rps: float
    output_tps: float
    total_tps: float
    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_itl_ms: float
    p95_e2e_ms: float
    mean_batch_size: float
    cache_hit_rate: float = 0.0


def generate_requests(
    count: int = 80,
    seed: int = 6,
    traffic: str = "mixed",
    repeated_prefixes: bool = False,
) -> list[Request]:
    rng = random.Random(seed)
    now = 0.0
    requests = []
    for idx in range(count):
        now += rng.expovariate(12.0) * 1000.0
        if traffic == "long_context":
            input_tokens = rng.randint(1200, 5000)
            output_tokens = rng.randint(32, 180)
        elif traffic == "decode_heavy":
            input_tokens = rng.randint(32, 256)
            output_tokens = rng.randint(256, 900)
        else:
            input_tokens = rng.randint(80, 1400)
            output_tokens = rng.randint(40, 360)
        prefix_id = idx % 5 if repeated_prefixes else None
        requests.append(Request(idx, now, input_tokens, output_tokens, prefix_id))
    return requests


# ---------------------------------------------------------------------------
# vLLM command builders


def vllm_serve_command(
    model: str = DEFAULT_MODEL,
    max_num_batched_tokens: int = 4096,
    max_num_seqs: int = 128,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    quantization: str | None = None,
    block_size: int | None = None,
    attention_backend: str | None = None,
) -> str:
    env = ""
    if attention_backend:
        env = (
            f"VLLM_ATTENTION_BACKEND={attention_backend} "
            "VLLM_USE_FLASHINFER_SAMPLER=1 "
            "VLLM_FLASHINFER_FORCE_TENSOR_CORES=1 "
        )
    parts = [
        f"{env}vllm serve {model}",
        f"--max-num-batched-tokens {max_num_batched_tokens}",
        f"--max-num-seqs {max_num_seqs}",
    ]
    if enable_chunked_prefill:
        parts.append("--enable-chunked-prefill")
    if enable_prefix_caching:
        parts.append("--enable-prefix-caching")
    if quantization:
        parts.append(f"--quantization {quantization}")
    if block_size:
        parts.append(f"--block-size {block_size}")
    return " \\\n  ".join(parts)


def vllm_bench_command(
    model: str = DEFAULT_MODEL,
    dataset_name: str = "sharegpt",
    request_rate: float = 10.0,
    max_concurrency: int = 64,
    num_prompts: int = 1000,
) -> str:
    parts = [
        "vllm bench serve",
        "--backend vllm",
        '--base-url "http://localhost:8000"',
        f"--model {model}",
        f"--dataset-name {dataset_name}",
        f"--num-prompts {num_prompts}",
        f"--request-rate {request_rate}",
        f"--max-concurrency {max_concurrency}",
        "--save-result",
        "--append-result",
        "--result-filename ch6_vllm_results.jsonl",
    ]
    if dataset_name == "prefix_repetition":
        parts += [
            "--prefix-repetition-prefix-len 512",
            "--prefix-repetition-suffix-len 128",
            "--prefix-repetition-num-prefixes 5",
            "--prefix-repetition-output-len 128",
        ]
    return " \\\n  ".join(parts)


# ---------------------------------------------------------------------------
# Batching simulations


def process_latency_ms(input_tokens: int, output_tokens: int, batch_size: int) -> float:
    prefill = 16.0 + input_tokens * 0.035
    decode = output_tokens * (6.0 / max(1.0, math.sqrt(batch_size)))
    return prefill + decode


def simulate_dynamic_batching(
    requests: list[Request],
    max_batch_size: int = 8,
    max_delay_ms: float = 20.0,
) -> ServingMetrics:
    pending: list[Request] = []
    now = 0.0
    idx = 0
    completion_times = {}
    first_token_times = {}
    itls = []
    batches = []

    while idx < len(requests) or pending:
        if not pending and idx < len(requests):
            now = max(now, requests[idx].arrival_ms)
            pending.append(requests[idx])
            idx += 1

        batch_deadline = pending[0].arrival_ms + max_delay_ms
        while idx < len(requests) and len(pending) < max_batch_size and requests[idx].arrival_ms <= batch_deadline:
            pending.append(requests[idx])
            idx += 1

        if len(pending) < max_batch_size and idx < len(requests) and now < batch_deadline:
            now = min(batch_deadline, requests[idx].arrival_ms)
            if requests[idx].arrival_ms <= now:
                continue

        batch = pending[:max_batch_size]
        pending = pending[max_batch_size:]
        batch_start = max(now, max(req.arrival_ms for req in batch))
        batch_size = len(batch)
        batch_work = max(process_latency_ms(req.input_tokens, req.output_tokens, batch_size) for req in batch)
        batch_finish = batch_start + batch_work
        batches.append(batch_size)
        for req in batch:
            first_token_times[req.request_id] = batch_start + 16.0 + req.input_tokens * 0.035
            completion_times[req.request_id] = batch_finish
            itls.append((batch_finish - first_token_times[req.request_id]) / max(req.output_tokens, 1))
        now = batch_finish

    return summarize_metrics("dynamic_batching", requests, first_token_times, completion_times, itls, batches)


def simulate_continuous_batching(
    requests: list[Request],
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 4096,
    chunked_prefill: bool = False,
    prefix_cache: bool = False,
) -> ServingMetrics:
    queued = list(requests)
    waiting: list[Request] = []
    active: list[dict[str, Any]] = []
    now = 0.0
    first_token_times = {}
    completion_times = {}
    itls = []
    batches = []
    prefix_cache_seen: OrderedDict[int, None] = OrderedDict()
    cache_hits = 0
    cache_lookups = 0

    while queued or waiting or active:
        while queued and queued[0].arrival_ms <= now:
            waiting.append(queued.pop(0))

        while waiting and len(active) < max_num_seqs:
            req = waiting.pop(0)
            input_tokens = req.input_tokens
            if prefix_cache and req.prefix_id is not None:
                cache_lookups += 1
                if req.prefix_id in prefix_cache_seen:
                    cache_hits += 1
                    input_tokens = max(16, int(input_tokens * 0.18))
                    prefix_cache_seen.move_to_end(req.prefix_id)
                else:
                    prefix_cache_seen[req.prefix_id] = None
                while len(prefix_cache_seen) > 16:
                    prefix_cache_seen.popitem(last=False)
            active.append(
                {
                    "req": req,
                    "prefill_left": input_tokens,
                    "decode_left": req.output_tokens,
                    "started_decode": False,
                    "last_token_time": None,
                }
            )

        if not active:
            if queued:
                now = queued[0].arrival_ms
            continue

        batch_token_budget = max_num_batched_tokens
        step_prefill = 0
        step_decode = 0
        for item in active:
            if item["prefill_left"] > 0:
                if chunked_prefill:
                    chunk = min(item["prefill_left"], max(64, batch_token_budget // max(len(active), 1)))
                else:
                    chunk = item["prefill_left"]
                chunk = min(chunk, batch_token_budget)
                item["prefill_left"] -= chunk
                batch_token_budget -= chunk
                step_prefill += chunk
            elif item["decode_left"] > 0:
                item["decode_left"] -= 1
                step_decode += 1
                if not item["started_decode"]:
                    item["started_decode"] = True
                    first_token_times[item["req"].request_id] = now
                elif item["last_token_time"] is not None:
                    itls.append(now - item["last_token_time"])
                item["last_token_time"] = now

        step_ms = 2.0
        if step_prefill:
            step_ms += step_prefill * 0.030
        if step_decode:
            step_ms += 6.0 / max(1.0, math.sqrt(step_decode))
        batches.append(len(active))
        now += step_ms

        still_active = []
        for item in active:
            if item["prefill_left"] <= 0 and item["decode_left"] <= 0:
                completion_times[item["req"].request_id] = now
            else:
                still_active.append(item)
        active = still_active

        while queued and queued[0].arrival_ms <= now:
            waiting.append(queued.pop(0))

    cache_hit_rate = cache_hits / cache_lookups if cache_lookups else 0.0
    name = "continuous_batching"
    if chunked_prefill:
        name += "+chunked_prefill"
    if prefix_cache:
        name += "+prefix_cache"
    return summarize_metrics(name, requests, first_token_times, completion_times, itls, batches, cache_hit_rate)


def summarize_metrics(
    strategy: str,
    requests: list[Request],
    first_token_times: dict[int, float],
    completion_times: dict[int, float],
    itls: list[float],
    batches: list[int],
    cache_hit_rate: float = 0.0,
) -> ServingMetrics:
    start = min(req.arrival_ms for req in requests)
    end = max(completion_times.values())
    duration_ms = max(1.0, end - start)
    ttfts = [first_token_times[req.request_id] - req.arrival_ms for req in requests]
    e2es = [completion_times[req.request_id] - req.arrival_ms for req in requests]
    total_in = sum(req.input_tokens for req in requests)
    total_out = sum(req.output_tokens for req in requests)
    return ServingMetrics(
        strategy=strategy,
        requests=len(requests),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        duration_ms=duration_ms,
        request_throughput_rps=len(requests) / (duration_ms / 1000.0),
        output_tps=total_out / (duration_ms / 1000.0),
        total_tps=(total_in + total_out) / (duration_ms / 1000.0),
        mean_ttft_ms=statistics.mean(ttfts),
        p95_ttft_ms=percentile(ttfts, 95),
        mean_itl_ms=statistics.mean(itls) if itls else 0.0,
        p95_e2e_ms=percentile(e2es, 95),
        mean_batch_size=statistics.mean(batches) if batches else 0.0,
        cache_hit_rate=cache_hit_rate,
    )


# ---------------------------------------------------------------------------
# Attention, PagedAttention, compression, prefix cache


def kv_cache_bytes(
    num_layers: int,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int,
    bytes_per_value: int = 2,
) -> int:
    # K and V are both stored.
    return num_layers * seq_len * num_kv_heads * head_dim * 2 * bytes_per_value


def attention_kv_comparison(
    seq_len: int = 32768,
    num_layers: int = 32,
    num_attention_heads: int = 32,
    head_dim: int = 128,
) -> dict[str, Any]:
    mha = kv_cache_bytes(num_layers, seq_len, head_dim, num_attention_heads)
    gqa = kv_cache_bytes(num_layers, seq_len, head_dim, 8)
    mqa = kv_cache_bytes(num_layers, seq_len, head_dim, 1)
    return {
        "config": {
            "seq_len": seq_len,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "head_dim": head_dim,
            "dtype": "fp16/bf16, 2 bytes",
        },
        "mha_gib": mha / 1024**3,
        "gqa_8kv_heads_gib": gqa / 1024**3,
        "mqa_1kv_head_gib": mqa / 1024**3,
        "gqa_memory_reduction_vs_mha": mha / gqa,
        "mqa_memory_reduction_vs_mha": mha / mqa,
        "chapter_concept": "Reducing KV heads with GQA/MQA lowers KV-cache memory and memory bandwidth pressure during decode.",
    }


def paged_attention_simulation(lengths: list[int] | None = None, block_size: int = 16) -> dict[str, Any]:
    if lengths is None:
        lengths = [17, 31, 128, 513, 1000, 2049, 4097]
    contiguous_reserved = sum(math.ceil(length / 512) * 512 for length in lengths)
    actual_tokens = sum(lengths)
    paged_reserved = sum(math.ceil(length / block_size) * block_size for length in lengths)
    return {
        "sequence_lengths": lengths,
        "actual_tokens": actual_tokens,
        "contiguous_reserved_tokens": contiguous_reserved,
        "contiguous_waste_percent": (contiguous_reserved - actual_tokens) / contiguous_reserved * 100,
        "paged_block_size": block_size,
        "paged_reserved_tokens": paged_reserved,
        "paged_waste_percent": (paged_reserved - actual_tokens) / paged_reserved * 100,
        "chapter_concept": "PagedAttention stores KV cache in fixed-size blocks to avoid large contiguous allocation waste.",
    }


def quantization_comparison(params_billions: float = 7.0) -> dict[str, Any]:
    formats = {
        "fp32": 4.0,
        "fp16/bf16": 2.0,
        "int8/fp8": 1.0,
        "int4/fp4": 0.5,
    }
    rows = {}
    for name, bytes_per_param in formats.items():
        rows[name] = {
            "weight_memory_gib": params_billions * 1_000_000_000 * bytes_per_param / 1024**3,
            "relative_to_fp16": bytes_per_param / formats["fp16/bf16"],
        }
    return {
        "params_billions": params_billions,
        "formats": rows,
        "chapter_concept": "Quantization reduces parameter precision to lower memory movement and increase room for KV cache.",
        "caution": "Lower precision can introduce rounding/clamping error; verify quality on your task.",
    }


def serving_metric_brief(metrics: ServingMetrics) -> dict[str, Any]:
    return {
        "strategy": metrics.strategy,
        "total_tps": round(metrics.total_tps, 2),
        "output_tps": round(metrics.output_tps, 2),
        "mean_ttft_ms": round(metrics.mean_ttft_ms, 2),
        "mean_itl_ms": round(metrics.mean_itl_ms, 4),
        "p95_e2e_ms": round(metrics.p95_e2e_ms, 2),
        "mean_batch_size": round(metrics.mean_batch_size, 2),
        "cache_hit_rate": round(metrics.cache_hit_rate, 4),
    }


def experiment_plan_table() -> list[dict[str, str]]:
    return [
        {
            "experiment": "調 max-num-seqs",
            "chapter_6_technique": "continuous batching",
            "local_section": "experiment-plan / batching",
        },
        {
            "experiment": "調 max-num-batched-tokens",
            "chapter_6_technique": "token-level batching",
            "local_section": "experiment-plan",
        },
        {
            "experiment": "開 / 關 chunked prefill",
            "chapter_6_technique": "chunked prefill",
            "local_section": "experiment-plan / chunked-prefill",
        },
        {
            "experiment": "原始模型 vs GPTQ / AWQ",
            "chapter_6_technique": "quantization",
            "local_section": "experiment-plan / quantization",
        },
        {
            "experiment": "原始模型 vs FP8",
            "chapter_6_technique": "W8A8 quantization",
            "local_section": "experiment-plan / quantization",
        },
        {
            "experiment": "default attention vs FlashInfer",
            "chapter_6_technique": "attention kernel optimization",
            "local_section": "experiment-plan / commands",
        },
    ]


def max_num_seqs_sweep(requests: list[Request], max_num_batched_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for max_num_seqs in [2, 4, 8, 16, 32]:
        metrics = simulate_continuous_batching(
            requests,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        rows.append(
            {
                "max_num_seqs": max_num_seqs,
                **serving_metric_brief(metrics),
                "serve_command": vllm_serve_command(
                    DEFAULT_MODEL,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                ),
            }
        )
    return rows


def max_num_batched_tokens_sweep(requests: list[Request], max_num_seqs: int) -> list[dict[str, Any]]:
    rows = []
    for max_num_batched_tokens in [1024, 2048, 4096, 8192, 16384]:
        metrics = simulate_continuous_batching(
            requests,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        rows.append(
            {
                "max_num_batched_tokens": max_num_batched_tokens,
                **serving_metric_brief(metrics),
                "serve_command": vllm_serve_command(
                    DEFAULT_MODEL,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                ),
            }
        )
    return rows


def chunked_prefill_experiment(requests: list[Request], max_num_seqs: int, max_num_batched_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for enabled in [False, True]:
        metrics = simulate_continuous_batching(
            requests,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            chunked_prefill=enabled,
        )
        rows.append(
            {
                "enable_chunked_prefill": enabled,
                **serving_metric_brief(metrics),
                "serve_command": vllm_serve_command(
                    DEFAULT_MODEL,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    enable_chunked_prefill=enabled,
                ),
            }
        )
    return rows


def quantized_model_experiment(params_billions: float = 7.0) -> list[dict[str, Any]]:
    fp16_gib = params_billions * 1_000_000_000 * 2.0 / 1024**3
    variants = [
        {
            "variant": "baseline_fp16",
            "model": DEFAULT_MODEL,
            "quantization_flag": None,
            "bytes_per_param": 2.0,
            "relative_throughput_hint": 1.0,
            "quality_risk": "baseline",
        },
        {
            "variant": "gptq_int4",
            "model": GPTQ_MODEL_PLACEHOLDER,
            "quantization_flag": "gptq",
            "bytes_per_param": 0.5,
            "relative_throughput_hint": 1.45,
            "quality_risk": "task-dependent; verify accuracy and formatting",
        },
        {
            "variant": "awq_int4",
            "model": DEFAULT_QUANT_MODEL,
            "quantization_flag": "awq",
            "bytes_per_param": 0.5,
            "relative_throughput_hint": 1.55,
            "quality_risk": "usually strong for instruction models, still verify",
        },
        {
            "variant": "fp8_w8a8",
            "model": FP8_MODEL_PLACEHOLDER,
            "quantization_flag": "fp8",
            "bytes_per_param": 1.0,
            "relative_throughput_hint": 1.25,
            "quality_risk": "hardware-dependent; activation ranges need validation",
        },
    ]
    rows = []
    for variant in variants:
        weight_memory_gib = params_billions * 1_000_000_000 * variant["bytes_per_param"] / 1024**3
        command = vllm_serve_command(
            variant["model"],
            max_num_batched_tokens=8192,
            max_num_seqs=128,
            quantization=variant["quantization_flag"],
        )
        rows.append(
            {
                "variant": variant["variant"],
                "weight_memory_gib": round(weight_memory_gib, 2),
                "relative_to_fp16_memory": round(weight_memory_gib / fp16_gib, 2),
                "relative_throughput_hint": variant["relative_throughput_hint"],
                "quality_risk": variant["quality_risk"],
                "serve_command": command,
            }
        )
    return rows


def attention_backend_experiment() -> list[dict[str, Any]]:
    return [
        {
            "backend": "default",
            "relative_throughput_hint": 1.0,
            "notes": "Use this as the baseline on your GPU and workload.",
            "serve_command": vllm_serve_command(DEFAULT_MODEL),
        },
        {
            "backend": "FlashInfer",
            "relative_throughput_hint": 1.08,
            "notes": "Often helps attention/sampling paths, but benefit depends on GPU, vLLM version, and sequence lengths.",
            "serve_command": vllm_serve_command(DEFAULT_MODEL, attention_backend="FLASHINFER"),
        },
    ]


def build_prompt(system: str, context_docs: list[str], user: str, tenant: str | None = None) -> str:
    tenant_line = f"Tenant: {tenant}\n" if tenant else ""
    docs = "\n".join(f"Document {idx + 1}: {doc}" for idx, doc in enumerate(context_docs))
    return f"System: {system}\n{tenant_line}Context:\n{docs}\nUser: {user}"


def common_prefix_tokens(a: str, b: str) -> int:
    a_tokens = a.split()
    b_tokens = b.split()
    count = 0
    for left, right in zip(a_tokens, b_tokens):
        if left != right:
            break
        count += 1
    return count


def prefix_cache_demo() -> dict[str, Any]:
    system = "You are a helpful assistant."
    docs = ["KV cache stores previous keys and values.", "Prefix caching reuses repeated prompt prefixes."]
    p1 = build_prompt(system, docs, "What is prefix caching?", tenant="customer-a")
    p2 = build_prompt(system, docs, "How does it affect TTFT?", tenant="customer-a")
    p3 = build_prompt(system, docs, "How does it affect TTFT?", tenant="customer-b")
    p4_bad = f"System: {system}\nTenant: customer-a\nContext:\nDocuments 1: {docs[0]}\nDocument 2: {docs[1]}\nUser: How does it affect TTFT?"
    return {
        "same_tenant_prefix_tokens": common_prefix_tokens(p1, p2),
        "different_tenant_prefix_tokens": common_prefix_tokens(p1, p3),
        "formatting_change_prefix_tokens": common_prefix_tokens(p1, p4_bad),
        "same_tenant_cache_hit": common_prefix_tokens(p1, p2) > 10,
        "different_tenant_isolated": common_prefix_tokens(p1, p3) < common_prefix_tokens(p1, p2),
        "formatting_warning": "Changing 'Document' to 'Documents' breaks the prefix earlier and can reduce cache hit length.",
        "chapter_concept": "Put static system/context first, dynamic user input last, and add tenant/session IDs to avoid cross-tenant prefix reuse.",
    }


def cache_aware_router(prefix: str, replicas: int = 4) -> int:
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % replicas


def cache_routing_demo() -> dict[str, Any]:
    prefixes = [
        "system helpful assistant context database docs",
        "system helpful assistant context database docs",
        "system helpful assistant context legal docs",
        "system helpful assistant context database docs",
        "system coding assistant context repository docs",
    ]
    routed = [cache_aware_router(prefix) for prefix in prefixes]
    return {
        "prefixes": prefixes,
        "replica_assignments": routed,
        "database_prefix_replicas": [routed[i] for i in [0, 1, 3]],
        "chapter_concept": "Cache-aware routing sends similar prefixes to the same replica so local KV prefix cache is reused.",
    }


# ---------------------------------------------------------------------------
# CLI demos


def demo_commands() -> None:
    print_json(
        {
            "example_6_1_batching": vllm_serve_command(
                DEFAULT_MODEL,
                max_num_batched_tokens=4096,
                max_num_seqs=128,
            ),
            "chunked_prefill": vllm_serve_command(
                DEFAULT_MODEL,
                max_num_batched_tokens=8192,
                max_num_seqs=128,
                enable_chunked_prefill=True,
            ),
            "prefix_caching": vllm_serve_command(
                DEFAULT_MODEL,
                max_num_batched_tokens=8192,
                max_num_seqs=128,
                enable_prefix_caching=True,
            ),
            "flashinfer_attention_backend": vllm_serve_command(
                DEFAULT_MODEL,
                attention_backend="FLASHINFER",
            ),
            "awq_quantized": vllm_serve_command(
                DEFAULT_QUANT_MODEL,
                quantization="awq",
                max_num_batched_tokens=8192,
                max_num_seqs=128,
                enable_prefix_caching=True,
            ),
            "gptq_quantized": vllm_serve_command(
                GPTQ_MODEL_PLACEHOLDER,
                quantization="gptq",
                max_num_batched_tokens=8192,
                max_num_seqs=128,
            ),
            "fp8_w8a8": vllm_serve_command(
                FP8_MODEL_PLACEHOLDER,
                quantization="fp8",
                max_num_batched_tokens=8192,
                max_num_seqs=128,
            ),
            "sharegpt_bench": vllm_bench_command(DEFAULT_MODEL, "sharegpt"),
            "prefix_repetition_bench": vllm_bench_command(DEFAULT_MODEL, "prefix_repetition"),
        }
    )


def demo_experiment_plan(args: argparse.Namespace) -> None:
    batching_requests = generate_requests(args.num_requests, traffic=args.traffic)
    long_context_requests = generate_requests(args.num_requests, traffic="long_context")
    print_json(
        {
            "experiment_to_chapter_concept": experiment_plan_table(),
            "max_num_seqs_sweep": max_num_seqs_sweep(
                batching_requests,
                max_num_batched_tokens=args.max_num_batched_tokens,
            ),
            "max_num_batched_tokens_sweep": max_num_batched_tokens_sweep(
                long_context_requests,
                max_num_seqs=args.max_num_seqs,
            ),
            "chunked_prefill_on_off": chunked_prefill_experiment(
                long_context_requests,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
            ),
            "baseline_vs_gptq_awq_fp8": quantized_model_experiment(),
            "default_attention_vs_flashinfer": attention_backend_experiment(),
            "how_to_read": [
                "Synthetic TPS/latency numbers show expected trade-off shape, not real GPU performance.",
                "For real results, run each serve_command on Linux/CUDA, then run vllm bench serve.",
                "Compare total TPS, output TPS, TTFT, ITL, P95 E2E latency, quality, and GPU memory together.",
            ],
        }
    )


def demo_batching(args: argparse.Namespace) -> None:
    requests = generate_requests(args.num_requests, traffic=args.traffic)
    results = [
        simulate_dynamic_batching(requests, args.max_num_seqs, args.max_delay_ms),
        simulate_continuous_batching(requests, args.max_num_seqs, args.max_num_batched_tokens),
    ]
    print_json([asdict(result) for result in results])


def demo_chunked_prefill(args: argparse.Namespace) -> None:
    requests = generate_requests(args.num_requests, traffic="long_context")
    results = [
        simulate_continuous_batching(
            requests,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            chunked_prefill=False,
        ),
        simulate_continuous_batching(
            requests,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            chunked_prefill=True,
        ),
    ]
    print_json(
        {
            "comparison": [asdict(result) for result in results],
            "interpretation": [
                "Chunked prefill is a trade-off, not a guaranteed win in every traffic pattern.",
                "It often improves decode smoothness/ITL by preventing long prefills from blocking decode.",
                "It may increase TTFT or lower throughput if chunks are too small or add too much overhead.",
                "Tune max-num-batched-tokens to avoid chunks that are too small or too large.",
            ],
        }
    )


def demo_attention() -> None:
    print_json(attention_kv_comparison())


def demo_paged_attention() -> None:
    print_json(paged_attention_simulation())


def demo_quantization() -> None:
    print_json(quantization_comparison())


def demo_prefix_cache(args: argparse.Namespace) -> None:
    requests = generate_requests(args.num_requests, traffic="long_context", repeated_prefixes=True)
    results = [
        simulate_continuous_batching(
            requests,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            chunked_prefill=True,
            prefix_cache=False,
        ),
        simulate_continuous_batching(
            requests,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            chunked_prefill=True,
            prefix_cache=True,
        ),
    ]
    print_json(
        {
            "prompt_formatting_demo": prefix_cache_demo(),
            "cache_aware_routing_demo": cache_routing_demo(),
            "synthetic_serving_comparison": [asdict(result) for result in results],
        }
    )


def demo_all(args: argparse.Namespace) -> None:
    for title, fn in [
        ("commands", lambda: demo_commands()),
        ("experiment-plan", lambda: demo_experiment_plan(args)),
        ("batching", lambda: demo_batching(args)),
        ("chunked-prefill", lambda: demo_chunked_prefill(args)),
        ("attention", demo_attention),
        ("paged-attention", demo_paged_attention),
        ("quantization", demo_quantization),
        ("prefix-cache", lambda: demo_prefix_cache(args)),
    ]:
        print(f"\n=== {title} ===")
        fn()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=[
            "commands",
            "experiment-plan",
            "batching",
            "chunked-prefill",
            "attention",
            "paged-attention",
            "quantization",
            "prefix-cache",
            "all",
        ],
        default="commands",
    )
    parser.add_argument("--num-requests", type=int, default=80)
    parser.add_argument("--traffic", choices=["mixed", "long_context", "decode_heavy"], default="mixed")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-delay-ms", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.section == "commands":
        demo_commands()
    elif args.section == "experiment-plan":
        demo_experiment_plan(args)
    elif args.section == "batching":
        demo_batching(args)
    elif args.section == "chunked-prefill":
        demo_chunked_prefill(args)
    elif args.section == "attention":
        demo_attention()
    elif args.section == "paged-attention":
        demo_paged_attention()
    elif args.section == "quantization":
        demo_quantization()
    elif args.section == "prefix-cache":
        demo_prefix_cache(args)
    elif args.section == "all":
        demo_all(args)


if __name__ == "__main__":
    main()
