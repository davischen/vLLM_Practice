# Chapter 6 vLLM Optimization Results and Explanations

This file records comparison outputs from
`ch6_vllm_optimization_techniques_hw.py` and explains which Chapter 6 concept
each result demonstrates.

The numbers here are synthetic local results. They are designed to show the
shape of each trade-off without requiring a CUDA machine. For real benchmarking,
use the `--section commands` output on a Linux/CUDA host with vLLM installed.

## Performance and Cost Summary

| Technique | Baseline | Optimized / comparison | Performance or cost lesson |
| --- | --- | --- | --- |
| Continuous batching | Dynamic batching: 192.26 output TPS, 41,310.82ms mean TTFT | Continuous batching: 1,378.58 output TPS, 1,677.17ms mean TTFT | Token-level scheduling greatly improves throughput and queue latency in the synthetic workload. |
| Chunked prefill | No chunking: 707.74 output TPS, 9.6219ms mean ITL | Chunked: 689.93 output TPS, 9.5259ms mean ITL | Chunking can smooth decode latency, but the chunk size must be tuned against throughput and TTFT. |
| GQA / MQA | MHA KV cache: 16 GiB | GQA: 4 GiB; MQA: 0.5 GiB | Fewer KV heads reduce memory cost and allow more concurrent cached tokens. |
| PagedAttention | Contiguous allocation wastes 27.13% token slots | Paged blocks waste 0.873% token slots | Fixed-size KV blocks reduce fragmentation and improve memory utilization. |
| Quantization | fp16/bf16 model memory: 13.04 GiB | int4/fp4 model memory: 3.26 GiB | Lower precision can cut weight memory to about 25% of fp16, with quality validation required. |
| Prefix caching | No cache: 19,795.72 total TPS, 1,295.75ms TTFT | Cache: 33,735.47 total TPS, 145.51ms TTFT | Reused prompt prefixes reduce prefill work and improve latency-sensitive serving. |
| `max-num-seqs` sweep | 2 seqs: 1,419.78 total TPS | 32 seqs: 8,874.65 total TPS | More concurrency raises throughput until another bottleneck or latency SLO becomes binding. |

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
python3 -B ch6_vllm_optimization_techniques_hw.py --section batching
```

Key comparison:

```text
dynamic_batching:
  requests: 80
  total_input_tokens: 61847
  total_output_tokens: 16722
  request_throughput_rps: 0.9198
  output_tps: 192.2599
  total_tps: 903.3409
  mean_ttft_ms: 41310.82
  p95_e2e_ms: 76798.17
  mean_batch_size: 1.1940

continuous_batching:
  requests: 80
  total_input_tokens: 61847
  total_output_tokens: 16722
  request_throughput_rps: 6.5953
  output_tps: 1378.5786
  total_tps: 6477.3077
  mean_ttft_ms: 1677.17
  p95_e2e_ms: 4461.73
  mean_batch_size: 7.1773
```

Chapter concept:

Dynamic batching waits for a batch window or batch size. Continuous batching
keeps the active batch full by adding new requests as soon as old ones finish.
This is why continuous batching is the default pattern for modern LLM serving.

Explanation:

In this synthetic mixed workload, continuous batching improves output token
throughput from `192.26 tok/s` to `1378.58 tok/s`, and total token throughput
from `903.34 tok/s` to `6477.31 tok/s`. Mean TTFT drops from about `41.31 s` to
`1.68 s`. The dynamic batching run has a low mean batch size because online
request arrivals do not always fill the batch before the delay deadline.

## Result 3: Chunked Prefill Trade-off

Command:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section chunked-prefill
```

Key comparison:

```text
continuous_batching:
  requests: 80
  total_input_tokens: 247070
  total_output_tokens: 8750
  output_tps: 707.7448
  total_tps: 20692.0318
  mean_ttft_ms: 2697.07
  mean_itl_ms: 9.6219
  p95_e2e_ms: 6089.20

continuous_batching+chunked_prefill:
  requests: 80
  total_input_tokens: 247070
  total_output_tokens: 8750
  output_tps: 689.9311
  total_tps: 20171.2195
  mean_ttft_ms: 2861.03
  mean_itl_ms: 9.5259
  p95_e2e_ms: 6356.76
```

Chapter concept:

Chunked prefill splits long prompt processing into smaller pieces so decode
steps are less likely to sit idle behind a long prefill. This can smooth ITL,
but it may increase TTFT or add overhead if the chunk size is poorly tuned.

Explanation:

This synthetic result shows the trade-off clearly: ITL improves slightly
(`9.6219 ms` to `9.5259 ms`), but TTFT, P95 E2E latency, and throughput regress
slightly. That does not mean chunked prefill is bad; it means chunk size and
workload shape matter.
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

```json
{
  "sequence_lengths": [
    17,
    31,
    128,
    513,
    1000,
    2049,
    4097
  ],
  "actual_tokens": 7835,
  "contiguous_reserved_tokens": 10752,
  "contiguous_waste_percent": 27.129836309523807,
  "paged_block_size": 16,
  "paged_reserved_tokens": 7904,
  "paged_waste_percent": 0.8729757085020243,
  "chapter_concept": "PagedAttention stores KV cache in fixed-size blocks to avoid large contiguous allocation waste."
}
```

Chapter concept:

PagedAttention stores KV cache in fixed-size blocks instead of requiring each
sequence to occupy one contiguous memory region. This reduces fragmentation and
lets vLLM pack variable-length requests much more efficiently.

Explanation:

The synthetic contiguous allocator reserves `10,752` token slots for `7,835`
actual tokens, wasting about `27.13%`. With 16-token PagedAttention blocks, it
reserves only `7,904` token slots and wastes about `0.873%`. This mirrors the
chapter's point that PagedAttention improves effective KV-cache utilization for
variable-length requests.

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
python3 -B ch6_vllm_optimization_techniques_hw.py --section experiment-plan --num-requests 40
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
  max_num_seqs=2:
    total_tps=1419.78, output_tps=301.63, mean_ttft_ms=11976.13
  max_num_seqs=4:
    total_tps=3286.67, output_tps=698.25, mean_ttft_ms=3726.56
  max_num_seqs=8:
    total_tps=6171.08, output_tps=1311.04, mean_ttft_ms=743.40
  max_num_seqs=16:
    total_tps=8291.25, output_tps=1761.47, mean_ttft_ms=80.82
  max_num_seqs=32:
    total_tps=8874.65, output_tps=1885.41, mean_ttft_ms=38.20

max_num_batched_tokens_sweep:
  1024:  total_tps=20020.56, output_tps=650.37, mean_ttft_ms=1267.41
  2048:  total_tps=20254.11, output_tps=657.95, mean_ttft_ms=1228.07
  4096:  total_tps=20375.71, output_tps=661.90, mean_ttft_ms=1208.99
  8192:  total_tps=20369.92, output_tps=661.72, mean_ttft_ms=1203.83
  16384: total_tps=20369.92, output_tps=661.72, mean_ttft_ms=1203.83

chunked_prefill_on_off:
  false: total_tps=20375.71, output_tps=661.90, mean_itl_ms=9.7388
  true:  total_tps=19795.72, output_tps=643.06, mean_itl_ms=9.6474

baseline_vs_gptq_awq_fp8:
  baseline_fp16: weight_memory_gib=13.04, relative_to_fp16_memory=1.0
  gptq_int4:    weight_memory_gib=3.26, relative_to_fp16_memory=0.25
  awq_int4:     weight_memory_gib=3.26, relative_to_fp16_memory=0.25
  fp8_w8a8:     weight_memory_gib=6.52, relative_to_fp16_memory=0.5

default_attention_vs_flashinfer:
  default backend relative_throughput_hint=1.0
  FlashInfer relative_throughput_hint=1.08
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

The sweep shows the tuning pattern clearly: increasing `max-num-seqs` improves
throughput until it starts to saturate around 16 to 32 active sequences, while
increasing `max-num-batched-tokens` has diminishing returns after 4096 to 8192
tokens in this synthetic long-context workload.
