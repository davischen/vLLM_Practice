# Chapter 9 LLM Optimization in Practice Results and Explanations

This file records outputs from `ch9_llm_optimization_practice_hw.py`. The local
results are synthetic, but they mirror the workflow from Chapter 9: inspect
hardware, inspect traffic, generate vLLM commands, run benchmarks, analyze
quantization, tune serving knobs, and decide whether distributed serving helps.

## Result 1: Hardware Inspection

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section hardware
```

Key output:

```json
{
  "available": false,
  "message": "nvidia-smi not found. This local lab can still run synthetic benchmarks.",
  "sample_expected_l40s": {
    "name": "NVIDIA L40S",
    "compute_cap": "8.9",
    "memory.total_mib": 46068,
    "memory.free_mib": 45469
  }
}
```

Chapter concept:

Optimization starts with hardware reality: GPU model, memory capacity, compute
capability, and whether the benchmark host matches the target deployment host.

Explanation:

On this machine, `nvidia-smi` is unavailable, so the lab falls back to synthetic
benchmarking. The sample L40S block shows what hardware metadata to record on a
real GPU host.

## Result 2: Dataset / Traffic Shape

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section dataset --dataset sharegpt
```

Key output:

```json
{
  "total_samples": 100,
  "prompt_length_distribution": {
    "min": 21,
    "max": 850,
    "mean": 201.7,
    "median": 125.5
  },
  "output_length_distribution": {
    "min": 10,
    "max": 800,
    "mean": 189.92,
    "median": 144.5
  }
}
```

Chapter concept:

Before tuning, inspect prompt lengths, output lengths, and traffic shape. Serving
knobs that help short prompts may not help long-context or decode-heavy traffic.

Explanation:

This synthetic ShareGPT-like workload has moderate prompts and varied output
lengths. That means both prefill and decode behavior matter.

## Result 3: vLLM Commands

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section commands
```

Key output:

```text
baseline_serve:
vllm serve Qwen/Qwen3-14B > vllm.log 2>&1 &

awq_serve:
vllm serve Qwen/Qwen3-14B-AWQ --quantization awq > vllm.log 2>&1 &

tuned_awq_serve:
vllm serve Qwen/Qwen3-14B-AWQ --quantization awq \
  --gpu-memory-utilization 0.95 \
  --max-model-len 1024 \
  --block-size 16 \
  --enable-prefix-caching \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill
```

Chapter concept:

Chapter 9 turns optimization ideas into repeatable benchmark commands: baseline,
quantized model, tuned serving config, ShareGPT benchmark, and prefix-repetition
benchmark.

Explanation:

Use these commands on a Linux/CUDA host with vLLM installed. The local script
prints them so the experiment plan is reproducible even when the current machine
cannot run vLLM.

## Result 4: Benchmark Comparison

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section benchmark
```

Key output:

```text
Qwen3-14B baseline, sharegpt:
  output_token_throughput_tps: 230.06
  total_token_throughput_tps: 474.38
  mean_ttft_ms: 93.78
  mean_itl_ms: 43.26

Qwen3-14B baseline, prefix_repetition:
  output_token_throughput_tps: 246.68
  total_token_throughput_tps: 1233.39
  mean_ttft_ms: 92.58
  mean_itl_ms: 37.49

Qwen3-14B-AWQ 4-bit, sharegpt:
  output_token_throughput_tps: 847.48
  total_token_throughput_tps: 1747.53
  mean_ttft_ms: 52.51
  mean_itl_ms: 23.42

Qwen3-14B-AWQ 4-bit, prefix_repetition:
  output_token_throughput_tps: 770.10
  total_token_throughput_tps: 3850.48
  mean_ttft_ms: 52.70
  mean_itl_ms: 20.81
```

Chapter concept:

Optimization should be measured with multiple metrics. Throughput, TTFT, ITL,
and traffic pattern can tell different stories.

Explanation:

The synthetic result shows AWQ improving total throughput and lowering latency
relative to the baseline. Prefix-repetition traffic also increases total token
throughput because repeated context is easier to reuse or amortize.

## Result 5: Quantization Analysis

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section quantization
```

Key output:

```json
{
  "baseline": {
    "model_memory_gib": 27.5185,
    "kv_cache_memory_gib": 11.0,
    "kv_cache_tokens": 72064,
    "max_concurrency_x": 1.76
  },
  "awq": {
    "model_memory_gib": 9.3619,
    "kv_cache_memory_gib": 29.15,
    "kv_cache_tokens": 191056,
    "max_concurrency_x": 4.66
  },
  "memory_saved_gib": 18.1566,
  "kv_cache_token_gain": 2.6512,
  "chapter_sharegpt_total_tps_gain": 2.7004
}
```

Chapter concept:

Quantization reduces weight memory, which can free GPU memory for KV cache and
increase serving concurrency.

Explanation:

The synthetic AWQ profile cuts model memory from about `27.52 GiB` to
`9.36 GiB`. That extra memory increases estimated KV-cache capacity from
`72,064` tokens to `191,056` tokens.

## Result 6: Tuning Sweep

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section tuning
```

Key output:

```text
Top profile by total TPS:
  awq_seqs=32_tokens=4096
  total_token_throughput_tps: 2602.92
  output_token_throughput_tps: 1262.31
  mean_ttft_ms: 70.01
  mean_itl_ms: 22.46

Homework:
  Change request-rate and max-concurrency; watch when TTFT grows.
  Change dataset to prefix_repetition; prefix caching should matter more.
  Pick the fastest config that still meets your latency SLO.
```

Chapter concept:

Serving optimization is a constrained search. You tune concurrency and token
budget, then choose the fastest config that still satisfies latency SLOs.

Explanation:

The top throughput config is not automatically the best product config. If TTFT
or tail latency violates the SLO, pick a lower-concurrency setting.

## Result 7: Distributed Serving Trade-offs

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section distributed
```

Key output:

```json
{
  "g6e_l40s_pcie": {
    "lesson": "Single-GPU can beat tensor parallelism when interconnect overhead is high.",
    "best_for": "horizontal scaling with multiple independent replicas"
  },
  "p4d_a100_nvlink": {
    "lesson": "NVLink makes tensor parallelism more useful for latency-oriented vertical scaling.",
    "four_gpu_tensor_parallel_tps": 3926,
    "four_independent_replicas_tps": 9816,
    "ttft_single_gpu_ms": 66,
    "ttft_four_gpu_ms": 33
  }
}
```

Chapter concept:

Distributed inference is not always faster. Tensor parallelism can reduce
per-request latency, but interconnect overhead can reduce aggregate throughput.

Explanation:

The result contrasts PCIe-style scaling with NVLink-style scaling. Horizontal
replicas are often simpler and higher-throughput when the model fits on one GPU.
Tensor parallelism is more attractive when the model does not fit or strict TTFT
is more important than total TPS.

## Result 8: Recommendation Helper

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section recommend --dataset prefix --latency-priority
```

Key output:

```json
{
  "traffic_profile": {
    "avg_prompt_tokens": 512,
    "avg_output_tokens": 128,
    "repeated_prefixes": true,
    "latency_priority": true,
    "single_gpu_fits": true
  },
  "recommendations": [
    "Enable prefix caching / LMCache-style optimizations...",
    "Limit concurrency and batch size; accept lower aggregate TPS for better TTFT.",
    "Try AWQ/4-bit quantization...",
    "Avoid overtuning..."
  ]
}
```

Chapter concept:

Optimization choices should follow the measured traffic profile and business
priority, not a generic checklist.

Explanation:

For repeated-prefix, latency-sensitive traffic, the helper recommends prefix
caching, conservative concurrency, AWQ validation, and a portable baseline.
