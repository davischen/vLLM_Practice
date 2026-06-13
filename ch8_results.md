# Chapter 8 LLM Serving Frameworks Results and Explanations

This file records outputs from `ch8_llm_serving_frameworks_hw.py`. The local
results are synthetic, but they mirror Chapter 8's serving-framework concepts:
vLLM architecture, initialization, request workflow, token-level scheduling,
framework trade-offs, and framework selection by workload.

## Performance and Cost Summary

| Experiment | Baseline / observation | Optimized / comparison | Performance or cost lesson |
| --- | --- | --- | --- |
| Request-level batching | 40 completed requests, 12,482.57 total TPS, 81.38ms mean TTFT | Simple batching has high synthetic TPS in this toy model | Whole-request batching is easy to reason about but does not expose vLLM's token-level control. |
| vLLM token-level scheduling | 40 completed requests, 11,254.12 total TPS, 209.72ms mean TTFT | Token budget is allocated step by step across active requests | Token scheduling separates request priority from how many tokens each request receives. |
| vLLM with chunked prefill and prefix cache | 40 completed requests, 11,285.92 total TPS, 61.55ms mean TTFT, 0.85 prefix-cache hit rate | Scheduled tokens drop from 71,208 to 51,474 | Prefix cache reduces repeated prefill work and improves TTFT in the synthetic long-prefill workload. |
| Scheduler demo | 10 completed requests, 2,532.17 total TPS, 438.51 output TPS | 0.5 prefix-cache hit rate and 2 preemptions | Priority scheduling can preempt lower-priority work, while prefix caching reuses previously computed prompt state. |
| Framework choice | vLLM, TensorRT-LLM, SGLang, and llama.cpp target different operating points | Structured-output profile ranks SGLang first, vLLM second | Choose by SLOs, workload shape, hardware, portability, and operational cost rather than a generic winner. |

## Result 1: vLLM Commands

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section commands
```

Key output:

```text
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-7B-Instruct",
    trust_remote_code=True,
    dtype="float16",
    max_model_len=32768,
    gpu_memory_utilization=0.8,
)

vllm serve Qwen/Qwen3-7B-Instruct \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

Chapter concept:

vLLM can be used either as an in-process Python library through `LLM`, or as an
OpenAI-compatible API server through `vllm serve`.

Explanation:

The command shows the two common integration modes from Chapter 8. Library mode
is useful for batch jobs or embedding vLLM inside another service. API-server
mode is better when many clients need a stable HTTP endpoint and streaming.

## Result 2: vLLM Architecture

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section architecture
```

Key output:

```json
{
  "chapter_concept": "vLLM separates public API, scheduling, execution, worker lifecycle, and model running.",
  "layers": [
    {"component": "LLMEngine", "responsibility": "Public entry point; receives generation requests and coordinates lifecycle."},
    {"component": "EngineCore", "responsibility": "Inner loop; asks Scheduler for work and passes outputs to processors."},
    {"component": "Scheduler", "responsibility": "Maintains WAITING/RUNNING queues, token budgets, KV blocks, and scheduling policy."},
    {"component": "ModelExecutor", "responsibility": "Orchestrates worker processes and dispatches SchedulerOutput for execution."},
    {"component": "GPUWorker", "responsibility": "Owns device state, communication, and model lifecycle in each worker process."},
    {"component": "GPUModelRunner", "responsibility": "Builds the actual model inputs and runs forward passes."}
  ]
}
```

Chapter concept:

vLLM is layered. The scheduler handles model-agnostic system decisions, while
execution workers and model runners handle architecture and hardware execution.

Explanation:

This separation keeps scheduling policies from being mixed with model-specific
kernel details. It also makes the framework easier to evolve as new models,
attention kernels, quantization formats, and hardware backends appear.

## Result 3: Multi-process Initialization

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section init
```

Key output:

```json
{
  "llm_constructor": {
    "model": "Qwen/Qwen3-7B-Instruct",
    "tensor_parallel_size": 4,
    "distributed_executor_backend": "mp"
  },
  "events": [
    {"step": 1, "actor": "main process", "event": "Create LLM config and initialize LLMEngine, EngineCore, Scheduler, KVCacheManager, and executor."},
    {"step": 2, "actor": "MultiProcessExecutor", "event": "Spawn 4 worker processes with distributed_executor_backend='mp'."},
    {"step": 3, "actor": "GPUWorker", "event": "Each worker sets CUDA device, initializes communication, and prepares response queues."},
    {"step": 4, "actor": "GPUModelRunner", "event": "Look up model implementation for Qwen/Qwen3-7B-Instruct and load sharded weights."}
  ]
}
```

Chapter concept:

In a multi-process vLLM setup, the main process initializes the serving
components and then coordinates worker processes that own device/model state.

Explanation:

This mirrors Chapter 8's initialization workflow: create the engine, spawn
workers, initialize each GPU worker, and load model shards through the model
runner.

## Result 4: Generation Workflow

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section workflow
```

Key output:

```json
{
  "workflow": [
    {"stage": "Processor", "output": "Validate inputs, tokenize prompts, create Request objects."},
    {"stage": "LLMEngine / EngineCore", "output": "Enter execution loop and ask Scheduler to build the next batch."},
    {
      "stage": "Scheduler",
      "output": {
        "scheduled_requests": ["req-1", "req-2"],
        "num_scheduled_tokens": {"req-1": 18, "req-2": 24},
        "metadata": ["KV block assignment", "attention metadata", "sampling metadata"]
      }
    },
    {"stage": "ModelExecutor", "output": "Broadcast SchedulerOutput to workers and run model forward pass."},
    {"stage": "OutputProcessor", "output": "Convert token IDs into streamed chunks or final RequestOutput objects."}
  ]
}
```

Chapter concept:

`SchedulerOutput` is the work order between scheduling and model execution.

Explanation:

The scheduler does not run the neural network directly. It decides what should
be run: which requests, how many tokens, and which metadata or KV-cache blocks
are needed. The executor consumes that work order and delegates the actual model
forward pass.

## Result 5: vLLM Scheduler Simulation

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section scheduler
```

Key output:

```json
{
  "config": {
    "max_num_seqs": 3,
    "max_num_batched_tokens": 384,
    "long_prefill_token_threshold": 128,
    "enable_prefix_cache": true,
    "preemption_mode": "recompute",
    "policy": "priority"
  },
  "summary": {
    "strategy": "vllm_token_level_scheduler",
    "requests": 10,
    "completed": 10,
    "duration_ms": 3144.734,
    "total_prompt_tokens": 6584,
    "total_output_tokens": 1379,
    "total_scheduled_tokens": 8767,
    "request_throughput_rps": 3.1799,
    "total_tps": 2532.1697,
    "output_tps": 438.5109,
    "mean_ttft_ms": 393.0952,
    "p95_ttft_ms": 950.8524,
    "mean_e2e_ms": 1046.2683,
    "p95_e2e_ms": 2136.0045,
    "prefix_cache_hit_rate": 0.5,
    "preemptions": 2
  },
  "first_scheduler_outputs": [
    {
      "step": 1,
      "scheduled_tokens": {"req-1": 128},
      "phase": {"req-1": "prefill"}
    },
    {
      "step": 7,
      "scheduled_tokens": {"req-1": 1},
      "phase": {"req-1": "decode"}
    }
  ]
}
```

Chapter concept:

vLLM schedules tokens, not whole requests. RUNNING requests are handled first,
WAITING requests are admitted when slots are available, and long prefills can be
split into smaller chunks.

Explanation:

The first scheduler outputs show `req-1` receiving 128 prefill tokens at a time
because `long_prefill_token_threshold` is set to `128`. Once the prompt is
computed, the same request moves to decode and receives one decode token per
step. The summary also shows priority-driven preemption and prefix-cache reuse.

## Result 6: Scheduler Comparison

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section compare-schedulers
```

Key output:

```text
request_level_batcher:
  completed: 40
  total_tps: 12482.5663
  output_tps: 712.0574
  mean_ttft_ms: 81.3797
  p95_ttft_ms: 130.6447
  total_scheduled_tokens: 71208

vllm_token_level_scheduler:
  completed: 40
  total_tps: 11254.1169
  output_tps: 641.9816
  mean_ttft_ms: 209.7188
  p95_ttft_ms: 455.4889
  prefix_cache_hit_rate: 0.0

vllm_token_level_scheduler + chunked prefill + prefix cache:
  completed: 40
  total_tps: 11285.9237
  output_tps: 643.7960
  mean_ttft_ms: 61.5510
  p95_ttft_ms: 114.2408
  total_scheduled_tokens: 51474
  prefix_cache_hit_rate: 0.85
```

Chapter concept:

Scheduling quality should be evaluated with more than throughput. TTFT, tail
latency, scheduled-token work, prefix-cache hit rate, and request lifecycle
behavior all matter.

Explanation:

The optimized token-level scheduler lowers mean TTFT from `209.72ms` to
`61.55ms` and lowers scheduled-token work from `71,208` to `51,474` by reusing
prefixes. In this synthetic model, the simple request-level batcher has higher
total TPS, but it does not model vLLM's fine-grained token budget decisions.

## Result 7: Framework Matrix

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section frameworks
```

Key output:

```json
[
  {
    "framework": "vLLM",
    "best_for": "Fast path to production, broad Hugging Face model support, OpenAI-compatible serving.",
    "strengths": "PagedAttention, continuous batching, prefix caching, speculative decoding, strong ecosystem.",
    "cost_profile": "Good datacenter GPU utilization with moderate operational complexity."
  },
  {
    "framework": "TensorRT-LLM",
    "best_for": "NVIDIA-centric production stacks chasing peak tokens per dollar.",
    "strengths": "TensorRT engines, FP8/FP4/INT4, in-flight batching, Triton/Dynamo integration.",
    "cost_profile": "High performance on NVIDIA hardware, higher build and engine-management complexity."
  },
  {
    "framework": "SGLang",
    "best_for": "Agentic, structured-output, multi-step, and multi-vendor workloads.",
    "strengths": "RadixAttention, structured generation, router, continuous batching, broad hardware targets.",
    "cost_profile": "Strong for complex serving flows; evaluate ecosystem and ops maturity for your team."
  },
  {
    "framework": "llama.cpp",
    "best_for": "Local, private, on-device, edge, and low-cost inference.",
    "strengths": "GGUF quantization, tiny dependency footprint, CPU/Metal/CUDA/Vulkan portability.",
    "cost_profile": "Lowest ops and infrastructure footprint, but not aimed at high-concurrency datacenter TPS."
  }
]
```

Chapter concept:

Serving-framework choice is contextual. The right tool depends on workload,
hardware, SLOs, portability, and operating model.

Explanation:

The matrix summarizes Chapter 8's practical framework comparison. vLLM is a
strong default for broad production serving, TensorRT-LLM targets NVIDIA peak
efficiency, SGLang is attractive for structured and agentic flows, and
llama.cpp is the low-footprint choice for local or edge inference.

## Result 8: Framework Recommendation Helper

Command:

```bash
python3 -B ch8_llm_serving_frameworks_hw.py --section evaluate --structured-outputs
```

Key output:

```json
{
  "profile": {
    "latency_priority": true,
    "high_throughput": true,
    "nvidia_only": false,
    "structured_outputs": true,
    "edge_or_private": false,
    "broad_model_support": true
  },
  "ranking": [
    {"framework": "SGLang", "score": 10},
    {"framework": "vLLM", "score": 8},
    {"framework": "TensorRT-LLM", "score": 5},
    {"framework": "llama.cpp", "score": 1}
  ]
}
```

Chapter concept:

Start from SLOs, workload shape, hardware reality, and operability, then choose
the framework that best matches those constraints.

Explanation:

With `--structured-outputs`, SGLang ranks first because the workload asks for
structured generation and multi-step serving strengths. vLLM remains second
because it has strong broad-model support and a production-friendly ecosystem.
