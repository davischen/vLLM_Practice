"""Chapter 9 LLM optimization-in-practice homework.

This file turns the chapter's optimization workflow into a runnable local lab.
It does not require a GPU or vLLM by default. Instead, it provides:

1. GPU hardware inspection with a safe fallback when nvidia-smi is unavailable.
2. ShareGPT-like and prefix-repetition synthetic benchmark traffic.
3. Dataset statistics and token-length histograms.
4. vLLM serve / vLLM bench command builders.
5. vLLM log parsing for model memory and KV-cache capacity.
6. Synthetic benchmark simulation for baseline, AWQ, cache-heavy, and tuning runs.
7. Quantization, prefix-cache, and distributed-serving trade-off analysis.

If you have the real environment, use --section commands to copy the generated
commands into a GPU machine. If not, use the synthetic sections to understand
the methodology and the expected shape of the results.

Examples:
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section hardware
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section dataset --dataset sharegpt
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section dataset --dataset prefix
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section commands
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section benchmark
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section quantization
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section tuning
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section distributed
    python3 -B examples/ch9_llm_optimization_practice_hw.py --section recommend
"""

# %%
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional


BASELINE_MODEL = "Qwen/Qwen3-14B"
AWQ_MODEL = "Qwen/Qwen3-14B-AWQ"


# %%
# Utility helpers


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


def histogram(values: list[int], bins: int = 10, width: int = 45) -> list[str]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [f"{low:4d}-{high:4d} tokens: " + "*" * width]
    bucket_size = (high - low + 1) / bins
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - low) / bucket_size))
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for idx, count in enumerate(counts):
        start = int(low + idx * bucket_size)
        end = int(low + (idx + 1) * bucket_size - 1)
        bar = "*" * max(1, round(width * count / max_count))
        lines.append(f"{start:4d}-{end:4d} tokens: {bar}")
    return lines


# %%
# Step 1. Hardware inspection


def inspect_nvidia_smi() -> dict[str, Any]:
    """Query NVIDIA GPU state if available; otherwise return an explanatory fallback."""

    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "message": "nvidia-smi not found. This local lab can still run synthetic benchmarks.",
            "sample_expected_l40s": {
                "name": "NVIDIA L40S",
                "compute_cap": "8.9",
                "memory.total_mib": 46068,
                "memory.used_mib": 0,
                "memory.free_mib": 45469,
            },
        }

    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.free,memory.used,memory.total,utilization.gpu,pstate,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    rows = []
    for line in output.strip().splitlines():
        name, compute_cap, free, used, total, util, pstate, power = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "name": name,
                "compute_cap": compute_cap,
                "memory.free_mib": int(free),
                "memory.used_mib": int(used),
                "memory.total_mib": int(total),
                "gpu_util_percent": int(util),
                "pstate": pstate,
                "power_watts": power,
            }
        )
    return {"available": True, "gpus": rows}


# %%
# Step 2. Generate / inspect benchmark traffic


@dataclass
class PromptRecord:
    request_id: str
    prompt: str
    prompt_len: int
    output_len: int
    prefix_id: Optional[int] = None


def words(count: int, seed: int) -> str:
    vocab = [
        "model",
        "serving",
        "latency",
        "throughput",
        "cache",
        "token",
        "prompt",
        "decode",
        "prefill",
        "batch",
        "request",
        "gpu",
        "memory",
        "traffic",
        "optimization",
    ]
    rng = random.Random(seed)
    return " ".join(rng.choice(vocab) for _ in range(count))


def generate_sharegpt_like(num_prompts: int = 100, seed: int = 7) -> list[PromptRecord]:
    """Generate conversational traffic with varied prompt/output lengths."""

    rng = random.Random(seed)
    records = []
    for idx in range(num_prompts):
        prompt_len = max(5, int(rng.lognormvariate(math.log(140), 0.9)))
        prompt_len = min(prompt_len, 850)
        output_len = max(4, int(rng.lognormvariate(math.log(160), 0.85)))
        output_len = min(output_len, 800)
        prompt = words(prompt_len, seed + idx)
        records.append(
            PromptRecord(
                request_id=str(idx + 1),
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )
    return records


def generate_prefix_repetition(
    num_prompts: int = 50,
    prefix_len: int = 256,
    suffix_len: int = 256,
    num_prefixes: int = 5,
    output_len: int = 128,
    seed: int = 11,
) -> list[PromptRecord]:
    """Synthetic cache-heavy traffic with repeated prompt prefixes."""

    prefixes = [words(prefix_len, seed + prefix_id) for prefix_id in range(num_prefixes)]
    records = []
    for idx in range(num_prompts):
        prefix_id = idx % num_prefixes
        suffix = words(suffix_len, seed * 10 + idx)
        prompt = prefixes[prefix_id] + " " + suffix
        records.append(
            PromptRecord(
                request_id=str(idx + 1),
                prompt=prompt,
                prompt_len=prefix_len + suffix_len,
                output_len=output_len,
                prefix_id=prefix_id,
            )
        )
    return records


def inspect_dataset(records: list[PromptRecord], save_samples: bool = False) -> dict[str, Any]:
    prompt_lens = [record.prompt_len for record in records]
    output_lens = [record.output_len for record in records]
    overview = {
        "total_samples": len(records),
        "prompt_length_distribution": {
            "min": min(prompt_lens),
            "max": max(prompt_lens),
            "mean": statistics.mean(prompt_lens),
            "median": statistics.median(prompt_lens),
            "std": statistics.pstdev(prompt_lens),
        },
        "output_length_distribution": {
            "min": min(output_lens),
            "max": max(output_lens),
            "mean": statistics.mean(output_lens),
            "median": statistics.median(output_lens),
            "std": statistics.pstdev(output_lens),
        },
        "prompt_length_histogram": histogram(prompt_lens),
        "sample_prompts": [
            {
                "request_id": record.request_id,
                "prompt_len": record.prompt_len,
                "output_len": record.output_len,
                "prefix_id": record.prefix_id,
                "prompt_preview": record.prompt[:120] + "...",
            }
            for record in records[:3]
        ],
    }
    if save_samples:
        with open("ch9_synthetic_samples.json", "w", encoding="utf-8") as f:
            json.dump([asdict(record) for record in records], f, indent=2)
        overview["saved_samples"] = "ch9_synthetic_samples.json"
    return overview


# %%
# Step 3. Metrics and synthetic benchmark simulation


@dataclass
class BenchmarkProfile:
    name: str
    model_weight_gib: float
    kv_cache_gib: float
    kv_cache_tokens: int
    base_ttft_ms: float
    base_itl_ms: float
    prefix_cache_multiplier: float
    memory_efficiency_multiplier: float


BASELINE_PROFILE = BenchmarkProfile(
    name="Qwen3-14B baseline",
    model_weight_gib=27.5185,
    kv_cache_gib=11.00,
    kv_cache_tokens=72064,
    base_ttft_ms=104.15,
    base_itl_ms=43.24,
    prefix_cache_multiplier=1.0,
    memory_efficiency_multiplier=1.0,
)

AWQ_PROFILE = BenchmarkProfile(
    name="Qwen3-14B-AWQ 4-bit",
    model_weight_gib=9.3619,
    kv_cache_gib=29.15,
    kv_cache_tokens=191056,
    base_ttft_ms=59.29,
    base_itl_ms=24.0,
    prefix_cache_multiplier=1.18,
    memory_efficiency_multiplier=2.7,
)


@dataclass
class BenchmarkResult:
    profile: str
    dataset: str
    successful_requests: int
    request_rate_configured: float
    benchmark_duration_s: float
    total_input_tokens: int
    total_generated_tokens: int
    request_throughput_rps: float
    output_token_throughput_tps: float
    total_token_throughput_tps: float
    mean_ttft_ms: float
    mean_itl_ms: float
    p99_itl_ms: float
    max_concurrency: int


def simulate_benchmark(
    records: list[PromptRecord],
    profile: BenchmarkProfile,
    dataset_name: str,
    request_rate: float,
    max_concurrency: int,
    seed: int = 13,
) -> BenchmarkResult:
    """Approximate vLLM benchmark behavior for local experimentation."""

    rng = random.Random(seed)
    total_input = sum(record.prompt_len for record in records)
    total_output = sum(record.output_len for record in records)
    repetition_factor = 1.0
    if any(record.prefix_id is not None for record in records):
        unique_prefixes = len({record.prefix_id for record in records})
        repetition_factor = max(1.0, len(records) / max(unique_prefixes, 1) / 2.0)

    cache_bonus = min(2.6, repetition_factor * profile.prefix_cache_multiplier)
    concurrency_bonus = min(1.0 + max_concurrency / 16.0, profile.memory_efficiency_multiplier)
    raw_total_tps = 474.38 * profile.memory_efficiency_multiplier * cache_bonus * (0.75 + 0.25 * concurrency_bonus)

    traffic_duration_floor = len(records) / max(request_rate, 0.001)
    compute_duration = (total_input + total_output) / raw_total_tps
    duration = max(traffic_duration_floor, compute_duration)

    itl_samples = [
        max(1.0, rng.gauss(profile.base_itl_ms / cache_bonus**0.15, profile.base_itl_ms * 0.18))
        for _ in range(min(total_output, 5000))
    ]
    mean_itl = statistics.mean(itl_samples) if itl_samples else profile.base_itl_ms
    p99_itl = percentile(itl_samples, 99) if itl_samples else profile.base_itl_ms
    mean_prompt_len = total_input / max(len(records), 1)
    mean_ttft = profile.base_ttft_ms * (0.85 + min(mean_prompt_len, 800) / 4000.0) / cache_bonus**0.10

    return BenchmarkResult(
        profile=profile.name,
        dataset=dataset_name,
        successful_requests=len(records),
        request_rate_configured=request_rate,
        benchmark_duration_s=duration,
        total_input_tokens=total_input,
        total_generated_tokens=total_output,
        request_throughput_rps=len(records) / duration,
        output_token_throughput_tps=total_output / duration,
        total_token_throughput_tps=(total_input + total_output) / duration,
        mean_ttft_ms=mean_ttft,
        mean_itl_ms=mean_itl,
        p99_itl_ms=p99_itl,
        max_concurrency=max_concurrency,
    )


# %%
# Step 4-7. vLLM command builders and log parsing


def build_vllm_serve_command(
    model: str = BASELINE_MODEL,
    tensor_parallel_size: Optional[int] = None,
    quantization: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    block_size: Optional[int] = None,
    enable_prefix_caching: bool = False,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    enable_chunked_prefill: bool = False,
) -> str:
    args = ["vllm", "serve", model]
    if tensor_parallel_size:
        args += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if quantization:
        args += ["--quantization", quantization]
    if gpu_memory_utilization:
        args += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_model_len:
        args += ["--max-model-len", str(max_model_len)]
    if block_size:
        args += ["--block-size", str(block_size)]
    if enable_prefix_caching:
        args.append("--enable-prefix-caching")
    if max_num_seqs:
        args += ["--max-num-seqs", str(max_num_seqs)]
    if max_num_batched_tokens:
        args += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    if enable_chunked_prefill:
        args.append("--enable-chunked-prefill")
    return " ".join(args) + " > vllm.log 2>&1 &"


def build_vllm_bench_command(
    dataset_name: str = "sharegpt",
    model: str = BASELINE_MODEL,
    num_prompts: int = 2000,
    request_rate: float = 10.0,
    max_concurrency: int = 10,
    dataset_path: Optional[str] = "ShareGPT_V3_unfiltered_cleaned_split.json",
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
        "--result-filename test_serve_results.txt",
    ]
    if dataset_name == "sharegpt" and dataset_path:
        parts.append(f"--dataset-path {dataset_path}")
    if dataset_name == "prefix_repetition":
        parts += [
            "--prefix-repetition-prefix-len 256",
            "--prefix-repetition-suffix-len 256",
            "--prefix-repetition-num-prefixes 10",
            "--prefix-repetition-output-len 128",
        ]
    return " \\\n  ".join(parts)


VLLM_LOG_SAMPLE = """
Loading weights took 4.47 seconds
Model loading took 27.5185 GiB and 5.265852 seconds
Available KV cache memory: 11.00 GiB
GPU KV cache size: 72,064 tokens
Maximum concurrency for 40,960 tokens per request: 1.76x
"""


def parse_vllm_log(log_text: str) -> dict[str, Any]:
    model_match = re.search(r"Model loading took\s+([0-9.]+)\s+GiB", log_text)
    kv_match = re.search(r"Available KV cache memory:\s+([0-9.]+)\s+GiB", log_text)
    tokens_match = re.search(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", log_text)
    concurrency_match = re.search(r"Maximum concurrency for\s+([0-9,]+)\s+tokens per request:\s+([0-9.]+)x", log_text)
    return {
        "model_memory_gib": float(model_match.group(1)) if model_match else None,
        "kv_cache_memory_gib": float(kv_match.group(1)) if kv_match else None,
        "kv_cache_tokens": int(tokens_match.group(1).replace(",", "")) if tokens_match else None,
        "max_concurrency_request_tokens": int(concurrency_match.group(1).replace(",", "")) if concurrency_match else None,
        "max_concurrency_x": float(concurrency_match.group(2)) if concurrency_match else None,
    }


# %%
# Step 8. Distributed serving analysis


def distributed_serving_analysis() -> dict[str, Any]:
    """Summarize the chapter's vertical vs horizontal scaling lesson."""

    return {
        "g6e_l40s_pcie": {
            "lesson": "Single-GPU can beat tensor parallelism when interconnect overhead is high.",
            "best_for": "horizontal scaling with multiple independent replicas",
            "recommended_start": build_vllm_serve_command(AWQ_MODEL, quantization="awq"),
            "caution": "L40S over PCIe lacks NVLink-style bandwidth, so 2/4 GPU tensor parallelism may reduce TPS.",
        },
        "p4d_a100_nvlink": {
            "lesson": "NVLink makes tensor parallelism more useful for latency-oriented vertical scaling.",
            "best_for": "large models or strict TTFT targets",
            "two_gpu_command": build_vllm_serve_command(AWQ_MODEL, tensor_parallel_size=2, quantization="awq"),
            "four_gpu_command": build_vllm_serve_command(AWQ_MODEL, tensor_parallel_size=4, quantization="awq"),
            "chapter_numbers": {
                "four_gpu_tensor_parallel_tps": 3926,
                "four_independent_replicas_tps": 9816,
                "ttft_single_gpu_ms": 66,
                "ttft_four_gpu_ms": 33,
            },
        },
        "rule_of_thumb": [
            "Use distributed serving when the model does not fit on one GPU.",
            "Use distributed serving when per-request latency matters more than aggregate throughput.",
            "Use horizontal replicas when throughput, simplicity, and fault isolation matter most.",
        ],
    }


def recommend_optimization(
    avg_prompt_tokens: int,
    avg_output_tokens: int,
    repeated_prefixes: bool,
    latency_priority: bool,
    single_gpu_fits: bool,
) -> list[str]:
    recs = []
    if not single_gpu_fits:
        recs.append("Use tensor parallel distributed serving; fitting the model is the first constraint.")
    if repeated_prefixes or avg_prompt_tokens > avg_output_tokens * 2:
        recs.append("Enable prefix caching / LMCache-style optimizations for prefill-heavy or repeated-prefix traffic.")
    if avg_output_tokens > avg_prompt_tokens * 2:
        recs.append("Evaluate speculative decoding for decode-heavy long-output traffic.")
    if latency_priority:
        recs.append("Limit concurrency and batch size; accept lower aggregate TPS for better TTFT.")
    else:
        recs.append("Increase max-num-seqs and max-num-batched-tokens until latency leaves the acceptable range.")
    recs.append("Try AWQ/4-bit quantization to reduce weight memory and expand KV-cache capacity, then verify quality.")
    recs.append("Avoid overtuning: keep a portable baseline and only specialize after measuring real traffic.")
    return recs


# %%
# CLI demos


def dataset_for_args(args: argparse.Namespace) -> list[PromptRecord]:
    if args.dataset == "sharegpt":
        return generate_sharegpt_like(args.num_prompts)
    return generate_prefix_repetition(
        num_prompts=args.num_prompts,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        num_prefixes=args.num_prefixes,
        output_len=args.output_len,
    )


def demo_dataset(args: argparse.Namespace) -> None:
    print_json(inspect_dataset(dataset_for_args(args), save_samples=args.save_samples))


def demo_commands(args: argparse.Namespace) -> None:
    tuned = build_vllm_serve_command(
        model=AWQ_MODEL,
        quantization="awq",
        gpu_memory_utilization=0.95,
        max_model_len=1024,
        block_size=16,
        enable_prefix_caching=True,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
    )
    print_json(
        {
            "baseline_serve": build_vllm_serve_command(BASELINE_MODEL),
            "awq_serve": build_vllm_serve_command(AWQ_MODEL, quantization="awq"),
            "tuned_awq_serve": tuned,
            "sharegpt_bench": build_vllm_bench_command("sharegpt", BASELINE_MODEL),
            "prefix_repetition_bench": build_vllm_bench_command(
                "prefix_repetition",
                BASELINE_MODEL,
                num_prompts=1000,
                request_rate=5,
            ),
        }
    )


def demo_benchmark(args: argparse.Namespace) -> None:
    sharegpt = generate_sharegpt_like(args.num_prompts)
    prefix = generate_prefix_repetition(args.num_prompts)
    results = [
        simulate_benchmark(sharegpt, BASELINE_PROFILE, "sharegpt", args.request_rate, args.max_concurrency),
        simulate_benchmark(prefix, BASELINE_PROFILE, "prefix_repetition", args.request_rate, args.max_concurrency),
        simulate_benchmark(sharegpt, AWQ_PROFILE, "sharegpt", args.request_rate, args.max_concurrency),
        simulate_benchmark(prefix, AWQ_PROFILE, "prefix_repetition", args.request_rate, args.max_concurrency),
    ]
    print_json([asdict(result) for result in results])


def demo_quantization(args: argparse.Namespace) -> None:
    baseline_log = parse_vllm_log(VLLM_LOG_SAMPLE)
    awq_log = parse_vllm_log(
        """
Model loading took 9.3619 GiB and 10.652314 seconds
Available KV cache memory: 29.15 GiB
GPU KV cache size: 191,056 tokens
Maximum concurrency for 40,960 tokens per request: 4.66x
"""
    )
    print_json(
        {
            "baseline": baseline_log,
            "awq": awq_log,
            "memory_saved_gib": baseline_log["model_memory_gib"] - awq_log["model_memory_gib"],
            "kv_cache_token_gain": awq_log["kv_cache_tokens"] / baseline_log["kv_cache_tokens"],
            "chapter_sharegpt_total_tps_gain": 1280 / 474,
            "chapter_ttft_improvement_percent": (103.61 - 59.29) / 103.61 * 100,
            "interpretation": "AWQ reduces weight memory, freeing GPU memory for a larger KV cache and higher concurrency.",
        }
    )


def demo_tuning(args: argparse.Namespace) -> None:
    records = generate_sharegpt_like(args.num_prompts)
    candidates = []
    for max_num_seqs in [4, 8, 16, 32]:
        for batched_tokens in [4096, 8192, 16384]:
            profile = BenchmarkProfile(
                name=f"awq_seqs={max_num_seqs}_tokens={batched_tokens}",
                model_weight_gib=AWQ_PROFILE.model_weight_gib,
                kv_cache_gib=AWQ_PROFILE.kv_cache_gib,
                kv_cache_tokens=AWQ_PROFILE.kv_cache_tokens,
                base_ttft_ms=AWQ_PROFILE.base_ttft_ms * (1 + max_num_seqs / 96),
                base_itl_ms=AWQ_PROFILE.base_itl_ms * (1 - min(0.20, batched_tokens / 100000)),
                prefix_cache_multiplier=AWQ_PROFILE.prefix_cache_multiplier,
                memory_efficiency_multiplier=min(3.1, 1.6 + max_num_seqs / 16 + batched_tokens / 20000),
            )
            result = simulate_benchmark(records, profile, "sharegpt", args.request_rate, max_num_seqs)
            candidates.append(asdict(result))
    candidates.sort(key=lambda item: (item["total_token_throughput_tps"], -item["mean_ttft_ms"]), reverse=True)
    print_json(
        {
            "top_5_by_total_tps": candidates[:5],
            "homework": [
                "Change request-rate and max-concurrency; watch when TTFT grows.",
                "Change dataset to prefix_repetition; prefix caching should matter more.",
                "Pick the fastest config that still meets your latency SLO.",
            ],
        }
    )


def demo_recommend(args: argparse.Namespace) -> None:
    records = dataset_for_args(args)
    avg_prompt = round(statistics.mean(record.prompt_len for record in records))
    avg_output = round(statistics.mean(record.output_len for record in records))
    repeated = any(record.prefix_id is not None for record in records)
    print_json(
        {
            "traffic_profile": {
                "avg_prompt_tokens": avg_prompt,
                "avg_output_tokens": avg_output,
                "repeated_prefixes": repeated,
                "latency_priority": args.latency_priority,
                "single_gpu_fits": args.single_gpu_fits,
            },
            "recommendations": recommend_optimization(
                avg_prompt,
                avg_output,
                repeated,
                args.latency_priority,
                args.single_gpu_fits,
            ),
        }
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=[
            "hardware",
            "dataset",
            "commands",
            "parse-log",
            "benchmark",
            "quantization",
            "tuning",
            "distributed",
            "recommend",
            "all",
        ],
        default="benchmark",
    )
    parser.add_argument("--dataset", choices=["sharegpt", "prefix"], default="sharegpt")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--suffix-len", type=int, default=256)
    parser.add_argument("--num-prefixes", type=int, default=5)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--latency-priority", action="store_true")
    parser.add_argument("--single-gpu-fits", action="store_true", default=True)
    return parser.parse_args(argv)


def run_section(args: argparse.Namespace, section: str) -> None:
    print(f"\n=== {section} ===")
    if section == "hardware":
        print_json(inspect_nvidia_smi())
    elif section == "dataset":
        demo_dataset(args)
    elif section == "commands":
        demo_commands(args)
    elif section == "parse-log":
        print_json(parse_vllm_log(VLLM_LOG_SAMPLE))
    elif section == "benchmark":
        demo_benchmark(args)
    elif section == "quantization":
        demo_quantization(args)
    elif section == "tuning":
        demo_tuning(args)
    elif section == "distributed":
        print_json(distributed_serving_analysis())
    elif section == "recommend":
        demo_recommend(args)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.section == "all":
        for section in [
            "hardware",
            "dataset",
            "commands",
            "parse-log",
            "benchmark",
            "quantization",
            "tuning",
            "distributed",
            "recommend",
        ]:
            run_section(args, section)
        return
    run_section(args, args.section)


if __name__ == "__main__":
    main()
