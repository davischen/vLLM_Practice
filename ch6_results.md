# Chapter 6 vLLM Optimization Results and Explanations

This file records comparison outputs from
`ch6_vllm_optimization_techniques_hw.py` and explains which Chapter 6 concept
each result demonstrates.

The numbers here are synthetic local results. They are designed to show the
shape of each trade-off without requiring a CUDA machine. For real benchmarking,
use the `--section commands` output on a Linux/CUDA host with vLLM installed.

## Result 1: vLLM Optimization Commands

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section commands
```

Key output:

```text
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --enable-chunked-prefill

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --enable-prefix-caching

VLLM_ATTENTION_BACKEND=FLASHINFER ... vllm serve Qwen/Qwen2.5-7B-Instruct

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --quantization awq

vllm serve YOUR_GPTQ_MODEL_ID \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --quantization gptq

vllm serve YOUR_FP8_MODEL_ID \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --quantization fp8
```

Chapter concept:

Chapter 6 introduces the main vLLM serving knobs for request scheduling,
chunked prefill, attention backend selection, prefix caching, and quantized
serving.

Explanation:

- `--max-num-seqs` caps how many requests can be active at once.
- `--max-num-batched-tokens` caps the total token work admitted into a batch.
- `--enable-chunked-prefill` splits long prefills into smaller chunks.
- `--enable-prefix-caching` keeps reusable prompt-prefix KV cache.
- `--quantization awq` uses a quantized model format to reduce model memory.
- `--quantization gptq` is for a GPTQ-quantized checkpoint.
- `--quantization fp8` is for FP8/W8A8 serving experiments on supported
  hardware/checkpoints.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` shows how to force an attention backend
  experiment.

## Result 2: Dynamic Batching vs Continuous Batching

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section batching --num-requests 40
```

Key comparison:

```text
dynamic_batching:
  request_throughput_rps: 0.91
  output_tps: 192.31
  total_tps: 905.19
  mean_ttft_ms: 21178.59
  p95_e2e_ms: 38520.79
  mean_batch_size: 1.21

continuous_batching:
  request_throughput_rps: 6.21
  output_tps: 1311.04
  total_tps: 6171.08
  mean_ttft_ms: 743.40
  p95_e2e_ms: 3020.56
  mean_batch_size: 6.89
```

Chapter concept:

Dynamic batching waits for a batch window or batch size. Continuous batching
keeps the active batch full by adding new requests as soon as old ones finish.
This is why continuous batching is the default pattern for modern LLM serving.

Explanation:

In this synthetic mixed workload, continuous batching improves output token
throughput from `192.31 tok/s` to `1311.04 tok/s`, while reducing mean TTFT from
about `21.18 s` to `0.74 s`. The dynamic batching run has a low mean batch size
because online request arrivals do not always fill the batch before the delay
deadline.

## Result 3: Chunked Prefill Trade-off

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section chunked-prefill --num-requests 40
```

Key comparison:

```text
continuous_batching:
  output_tps: 661.90
  total_tps: 20375.71
  mean_ttft_ms: 1208.99
  mean_itl_ms: 9.7388

continuous_batching+chunked_prefill:
  output_tps: 643.06
  total_tps: 19795.72
  mean_ttft_ms: 1295.75
  mean_itl_ms: 9.6474
```

Chapter concept:

Chunked prefill splits long prompt processing into smaller pieces so decode
steps are less likely to sit idle behind a long prefill. This can smooth ITL,
but it may increase TTFT or add overhead if the chunk size is poorly tuned.

Explanation:

This synthetic result shows the trade-off clearly: ITL improves slightly
(`9.7388 ms` to `9.6474 ms`), but TTFT and throughput regress slightly. That does
not mean chunked prefill is bad; it means chunk size and workload shape matter.
For long-context interactive workloads, you should tune
`--max-num-batched-tokens` and check TTFT, ITL, and E2E latency together.

## Result 4: MHA vs GQA vs MQA KV Cache

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section attention
```

Key output:

```text
seq_len: 32768
num_layers: 32
num_attention_heads: 32
head_dim: 128

mha_gib: 16.0
gqa_8kv_heads_gib: 4.0
mqa_1kv_head_gib: 0.5
gqa_memory_reduction_vs_mha: 4.0
mqa_memory_reduction_vs_mha: 32.0
```

Chapter concept:

MHA stores KV for every attention head. GQA shares KV heads across groups of
query heads. MQA shares one KV head across all query heads. Fewer KV heads mean
less KV-cache memory and less memory bandwidth pressure during decode.

Explanation:

For this 32-layer, 32-head, 32k-token example, MHA needs about `16 GiB` of KV
cache, while GQA with 8 KV heads needs `4 GiB`, and MQA needs only `0.5 GiB`.
This is why model architecture choices matter for serving, not only for quality.

## Result 5: PagedAttention Memory Waste

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section paged-attention
```

Key output:

```text
actual_tokens: 7835
contiguous_reserved_tokens: 10752
contiguous_waste_percent: 27.13
paged_block_size: 16
paged_reserved_tokens: 7904
paged_waste_percent: 0.87
```

Chapter concept:

PagedAttention stores KV cache in fixed-size blocks instead of requiring each
sequence to occupy one contiguous memory region. This reduces fragmentation and
lets vLLM pack variable-length requests much more efficiently.

Explanation:

The synthetic contiguous allocator wastes about `27.13%` of reserved token
slots. The paged allocator wastes only `0.87%`. This mirrors the chapter's
point that PagedAttention improves effective KV-cache utilization.

## Result 6: Quantization Memory Savings

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section quantization
```

Key output:

```text
7B model weight memory estimate:
  fp32:      26.08 GiB
  fp16/bf16: 13.04 GiB
  int8/fp8:   6.52 GiB
  int4/fp4:   3.26 GiB
```

Chapter concept:

Quantization reduces parameter precision, shrinking model weights and lowering
memory movement. In serving, this can increase throughput and leave more GPU
memory available for KV cache.

Explanation:

Moving from FP16/BF16 to INT4/FP4 cuts estimated weight memory to about one
quarter of FP16. The trade-off is quality risk from rounding or clamping error,
so any quantized model should be validated against the task's accuracy or
preference benchmark.

## Result 7: Prefix Caching and Cache-aware Routing

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section prefix-cache --num-requests 40
```

Key output:

```text
same_tenant_prefix_tokens: 27
different_tenant_prefix_tokens: 7
formatting_change_prefix_tokens: 9
same_tenant_cache_hit: true
different_tenant_isolated: true
cache_hit_rate: 0.875

without prefix cache:
  output_tps: 643.06
  total_tps: 19795.72
  mean_ttft_ms: 1295.75
  p95_e2e_ms: 3736.06

with prefix cache:
  output_tps: 1095.90
  total_tps: 33735.47
  mean_ttft_ms: 145.51
  p95_e2e_ms: 1090.84
```

Chapter concept:

Prefix caching reuses KV cache for repeated prompt prefixes. It is especially
useful for multiturn chat and long-context prompts where the system prompt or
documents stay stable while the user query changes.

Explanation:

The prompt-formatting demo shows three practical lessons:

- Stable formatting produces a long shared prefix.
- Adding a tenant/session ID reduces accidental cross-tenant sharing.
- Small formatting changes, such as `Document` vs `Documents`, can shorten the
  prefix match and reduce cache benefit.

The synthetic serving comparison shows why prefix caching matters: with a cache
hit rate of `0.875`, mean TTFT drops from about `1295.75 ms` to `145.51 ms`, and
total token throughput rises from `19795.72 tok/s` to `33735.47 tok/s`.

## Result 8: Chapter 6 Experiment Plan Matrix

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section experiment-plan --num-requests 20
```

Experiment-to-concept mapping:

| 實驗 | 對應 Chapter 6 技術 |
| --- | --- |
| 調 `max-num-seqs` | continuous batching |
| 調 `max-num-batched-tokens` | token-level batching |
| 開 / 關 chunked prefill | chunked prefill |
| 原始模型 vs GPTQ / AWQ | quantization |
| 原始模型 vs FP8 | W8A8 quantization |
| default attention vs FlashInfer | attention kernel optimization |

Key output shape:

```text
experiment_to_chapter_concept:
  調 max-num-seqs -> continuous batching
  調 max-num-batched-tokens -> token-level batching
  開 / 關 chunked prefill -> chunked prefill
  原始模型 vs GPTQ / AWQ -> quantization
  原始模型 vs FP8 -> W8A8 quantization
  default attention vs FlashInfer -> attention kernel optimization

max_num_seqs_sweep:
  max_num_seqs=2, 4, 8, 16, 32
  compare total_tps, output_tps, mean_ttft_ms, mean_itl_ms, p95_e2e_ms

max_num_batched_tokens_sweep:
  max_num_batched_tokens=1024, 2048, 4096, 8192, 16384
  compare token-level batching budget effects

chunked_prefill_on_off:
  enable_chunked_prefill=false vs true

baseline_vs_gptq_awq_fp8:
  baseline_fp16, gptq_int4, awq_int4, fp8_w8a8

default_attention_vs_flashinfer:
  default backend vs VLLM_ATTENTION_BACKEND=FLASHINFER
```

Chapter concept:

This section acts as the Chapter 6 homework checklist. It turns the chapter's
optimization techniques into a concrete benchmark matrix: scheduler capacity,
token budget, prefill policy, quantized model format, and attention kernel.

Explanation:

The local values are synthetic and should be treated as a planning aid. For real
measurements, run each generated `serve_command` on a Linux/CUDA machine with
vLLM, then run the generated `vllm bench serve` command against it. Record at
least total TPS, output TPS, TTFT, ITL, P95 E2E latency, GPU memory, and any
quality/regression notes.
