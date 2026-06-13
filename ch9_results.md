# Chapter 9 LLM Optimization in Practice Results and Explanations

This file records outputs from `ch9_llm_optimization_practice_hw.py`. The local
results are synthetic, but they mirror the workflow from Chapter 9: inspect
hardware, inspect traffic, generate vLLM commands, parse logs, run benchmarks,
analyze quantization, tune serving knobs, and decide whether distributed serving
helps.

## Performance and Cost Summary

| Experiment | Baseline | Optimized / comparison | Performance or cost lesson |
| --- | --- | --- | --- |
| ShareGPT traffic | Baseline: 474.38 total TPS, 93.78ms TTFT, 43.26ms ITL | AWQ: 1,747.53 total TPS, 52.51ms TTFT, 23.42ms ITL | AWQ is about 3.68x higher total TPS in the synthetic ShareGPT workload. |
| Prefix-repetition traffic | Baseline: 1,233.39 total TPS, 92.58ms TTFT | AWQ: 3,850.48 total TPS, 52.70ms TTFT | Repeated-prefix workloads benefit from both reduced weight cost and reusable context. |
| Quantization memory | Baseline model: 27.5185 GiB; KV cache: 11.0 GiB | AWQ model: 9.3619 GiB; KV cache: 29.15 GiB | Saving 18.1566 GiB of model memory increases estimated KV capacity by about 2.65x. |
| Tuning sweep | Untuned profiles vary by concurrency and token budget | Best shown profile: 2,602.92 total TPS, 70.01ms TTFT | Pick the fastest config that still satisfies latency SLOs, not only the highest TPS. |
| Distributed serving | 4-GPU tensor parallel: 3,926 TPS, TTFT 33ms | Four independent replicas: 9,816 TPS, single-GPU TTFT 66ms | Tensor parallelism can reduce per-request latency; replicas can win aggregate throughput. |
| Recommendation helper | Generic tuning checklist | Traffic-aware advice for prefix-heavy, latency-sensitive serving | Optimization choices should follow workload shape and business priority. |

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
    "memory.used_mib": 0,
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

## Result 2: ShareGPT-like Traffic Shape

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
    "median": 125.5,
    "std": 203.15853415497958
  },
  "output_length_distribution": {
    "min": 10,
    "max": 800,
    "mean": 189.92,
    "median": 144.5,
    "std": 163.91953391832226
  },
  "prompt_length_histogram": [
    "  21- 103 tokens: *********************************************",
    " 104- 186 tokens: *****************************",
    " 187- 269 tokens: **************",
    " 270- 352 tokens: ********",
    " 353- 435 tokens: ****",
    " 436- 518 tokens: **",
    " 519- 601 tokens: *",
    " 602- 684 tokens: *",
    " 685- 767 tokens: **",
    " 768- 850 tokens: ******"
  ]
}
```

Chapter concept:

Before tuning, inspect prompt lengths, output lengths, and traffic shape. Serving
knobs that help short prompts may not help long-context or decode-heavy traffic.

Explanation:

This synthetic ShareGPT-like workload has varied prompt and output lengths. That
means both prefill and decode behavior matter, and one metric cannot explain the
whole serving profile.

## Result 3: Prefix-repetition Traffic Shape

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section dataset --dataset prefix
```

Key output:

```json
{
  "total_samples": 100,
  "prompt_length_distribution": {
    "min": 512,
    "max": 512,
    "mean": 512,
    "median": 512.0,
    "std": 0.0
  },
  "output_length_distribution": {
    "min": 128,
    "max": 128,
    "mean": 128,
    "median": 128.0,
    "std": 0.0
  },
  "prompt_length_histogram": [
    " 512- 512 tokens: *********************************************"
  ],
  "sample_prompts": [
    {
      "request_id": "1",
      "prompt_len": 512,
      "output_len": 128,
      "prefix_id": 0
    },
    {
      "request_id": "2",
      "prompt_len": 512,
      "output_len": 128,
      "prefix_id": 1
    },
    {
      "request_id": "3",
      "prompt_len": 512,
      "output_len": 128,
      "prefix_id": 2
    }
  ]
}
```

Chapter concept:

Prefix-heavy workloads are designed to test prefix caching, prefill reuse, and
long-context serving behavior.

Explanation:

All prompts are exactly `512` tokens and all outputs are `128` tokens. The
presence of `prefix_id` indicates repeated-prefix traffic, which is where prefix
caching or LMCache-style optimizations should matter more.

## Result 4: vLLM Commands

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
vllm serve Qwen/Qwen3-14B-AWQ --quantization awq --gpu-memory-utilization 0.95 --max-model-len 1024 --block-size 16 --enable-prefix-caching --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill > vllm.log 2>&1 &

sharegpt_bench:
vllm bench serve \
  --backend vllm \
  --base-url "http://localhost:8000" \
  --model Qwen/Qwen3-14B \
  --dataset-name sharegpt \
  --num-prompts 2000 \
  --request-rate 10.0 \
  --max-concurrency 10 \
  --save-result \
  --append-result \
  --result-filename test_serve_results.txt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

prefix_repetition_bench:
vllm bench serve \
  --backend vllm \
  --base-url "http://localhost:8000" \
  --model Qwen/Qwen3-14B \
  --dataset-name prefix_repetition \
  --num-prompts 1000 \
  --request-rate 5 \
  --max-concurrency 10 \
  --save-result \
  --append-result \
  --result-filename test_serve_results.txt \
  --prefix-repetition-prefix-len 256 \
  --prefix-repetition-suffix-len 256 \
  --prefix-repetition-num-prefixes 10 \
  --prefix-repetition-output-len 128
```

Chapter concept:

Chapter 9 turns optimization ideas into repeatable benchmark commands: baseline,
quantized model, tuned serving config, ShareGPT benchmark, and prefix-repetition
benchmark.

Explanation:

Use these commands on a Linux/CUDA host with vLLM installed. The local script
prints them so the experiment plan is reproducible even when the current machine
cannot run vLLM.

## Result 5: Parse vLLM Log

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section parse-log
```

Key output:

```json
{
  "model_memory_gib": 27.5185,
  "kv_cache_memory_gib": 11.0,
  "kv_cache_tokens": 72064,
  "max_concurrency_request_tokens": 40960,
  "max_concurrency_x": 1.76
}
```

Chapter concept:

vLLM logs expose model memory and KV-cache capacity. Those numbers help explain
why a serving configuration can or cannot support a target concurrency.

Explanation:

This parsed baseline profile estimates `27.52 GiB` for model weights and
`11.0 GiB` for KV cache, enough for about `72,064` KV-cache tokens. For a
request budget of `40,960` tokens, that is about `1.76x` concurrency headroom.

## Result 6: Benchmark Comparison

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section benchmark
```

Key output:

```text
Qwen3-14B baseline, sharegpt:
  successful_requests: 100
  request_throughput_rps: 1.2113
  output_token_throughput_tps: 230.06
  total_token_throughput_tps: 474.38
  mean_ttft_ms: 93.78
  mean_itl_ms: 43.26
  p99_itl_ms: 61.29

Qwen3-14B baseline, prefix_repetition:
  successful_requests: 100
  request_throughput_rps: 1.9272
  output_token_throughput_tps: 246.68
  total_token_throughput_tps: 1233.39
  mean_ttft_ms: 92.58
  mean_itl_ms: 37.49
  p99_itl_ms: 55.51

Qwen3-14B-AWQ 4-bit, sharegpt:
  successful_requests: 100
  request_throughput_rps: 4.4623
  output_token_throughput_tps: 847.48
  total_token_throughput_tps: 1747.53
  mean_ttft_ms: 52.51
  mean_itl_ms: 23.42
  p99_itl_ms: 33.43

Qwen3-14B-AWQ 4-bit, prefix_repetition:
  successful_requests: 100
  request_throughput_rps: 6.0164
  output_token_throughput_tps: 770.10
  total_token_throughput_tps: 3850.48
  mean_ttft_ms: 52.70
  mean_itl_ms: 20.81
  p99_itl_ms: 30.81
```

Chapter concept:

Optimization should be measured with multiple metrics. Throughput, TTFT, ITL,
P99 ITL, and traffic pattern can tell different stories.

Explanation:

The synthetic result shows AWQ improving total throughput and lowering latency
relative to the baseline. Prefix-repetition traffic also increases total token
throughput because repeated context is easier to reuse or amortize.

## Result 7: Quantization Analysis

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
    "max_concurrency_request_tokens": 40960,
    "max_concurrency_x": 1.76
  },
  "awq": {
    "model_memory_gib": 9.3619,
    "kv_cache_memory_gib": 29.15,
    "kv_cache_tokens": 191056,
    "max_concurrency_request_tokens": 40960,
    "max_concurrency_x": 4.66
  },
  "memory_saved_gib": 18.156599999999997,
  "kv_cache_token_gain": 2.6511989342806395,
  "chapter_sharegpt_total_tps_gain": 2.70042194092827,
  "chapter_ttft_improvement_percent": 42.77579384229321,
  "interpretation": "AWQ reduces weight memory, freeing GPU memory for a larger KV cache and higher concurrency."
}
```

Chapter concept:

Quantization reduces weight memory, which can free GPU memory for KV cache and
increase serving concurrency.

Explanation:

The synthetic AWQ profile cuts model memory from about `27.52 GiB` to
`9.36 GiB`. That extra memory increases estimated KV-cache capacity from
`72,064` tokens to `191,056` tokens, and the synthetic ShareGPT total TPS gain
is about `2.70x`.

## Result 8: Tuning Sweep

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section tuning
```

Key output:

```text
Top 5 by total TPS:
1. awq_seqs=32_tokens=4096
   total_token_throughput_tps: 2602.92
   output_token_throughput_tps: 1262.31
   request_throughput_rps: 6.6466
   mean_ttft_ms: 70.01
   mean_itl_ms: 22.46
   p99_itl_ms: 32.06

2. awq_seqs=32_tokens=8192
   total_token_throughput_tps: 2602.92
   output_token_throughput_tps: 1262.31
   mean_ttft_ms: 70.01
   mean_itl_ms: 21.50
   p99_itl_ms: 30.69

3. awq_seqs=32_tokens=16384
   total_token_throughput_tps: 2602.92
   output_token_throughput_tps: 1262.31
   mean_ttft_ms: 70.01
   mean_itl_ms: 19.58
   p99_itl_ms: 27.95
```

Homework prompts:

```text
Change request-rate and max-concurrency; watch when TTFT grows.
Change dataset to prefix_repetition; prefix caching should matter more.
Pick the fastest config that still meets your latency SLO.
```

Chapter concept:

Serving optimization is a constrained search. You tune concurrency and token
budget, then choose the fastest config that still satisfies latency SLOs.

Explanation:

The top throughput configs use AWQ with `max_num_seqs=32`. Larger token budgets
do not change total TPS in this synthetic sweep, but they improve ITL/P99 ITL.
The fastest config is not automatically the best product config if TTFT or tail
latency violates the SLO.

## Result 9: Distributed Serving Trade-offs

Command:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section distributed
```

Key output:

```json
{
  "g6e_l40s_pcie": {
    "lesson": "Single-GPU can beat tensor parallelism when interconnect overhead is high.",
    "best_for": "horizontal scaling with multiple independent replicas",
    "recommended_start": "vllm serve Qwen/Qwen3-14B-AWQ --quantization awq > vllm.log 2>&1 &",
    "caution": "L40S over PCIe lacks NVLink-style bandwidth, so 2/4 GPU tensor parallelism may reduce TPS."
  },
  "p4d_a100_nvlink": {
    "lesson": "NVLink makes tensor parallelism more useful for latency-oriented vertical scaling.",
    "best_for": "large models or strict TTFT targets",
    "two_gpu_command": "vllm serve Qwen/Qwen3-14B-AWQ --tensor-parallel-size 2 --quantization awq > vllm.log 2>&1 &",
    "four_gpu_command": "vllm serve Qwen/Qwen3-14B-AWQ --tensor-parallel-size 4 --quantization awq > vllm.log 2>&1 &",
    "chapter_numbers": {
      "four_gpu_tensor_parallel_tps": 3926,
      "four_independent_replicas_tps": 9816,
      "ttft_single_gpu_ms": 66,
      "ttft_four_gpu_ms": 33
    }
  },
  "rule_of_thumb": [
    "Use distributed serving when the model does not fit on one GPU.",
    "Use distributed serving when per-request latency matters more than aggregate throughput.",
    "Use horizontal replicas when throughput, simplicity, and fault isolation matter most."
  ]
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

## Result 10: Recommendation Helper

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
    "Enable prefix caching / LMCache-style optimizations for prefill-heavy or repeated-prefix traffic.",
    "Limit concurrency and batch size; accept lower aggregate TPS for better TTFT.",
    "Try AWQ/4-bit quantization to reduce weight memory and expand KV-cache capacity, then verify quality.",
    "Avoid overtuning: keep a portable baseline and only specialize after measuring real traffic."
  ]
}
```

Chapter concept:

Optimization choices should follow the measured traffic profile and business
priority, not a generic checklist.

Explanation:

For repeated-prefix, latency-sensitive traffic, the helper recommends prefix
caching, conservative concurrency, AWQ validation, and a portable baseline.
