# LLM Model Serving Homework Examples

This folder contains runnable Python homework/lab programs for the book chapters.
Most examples are designed to run locally with deterministic mock backends, so
you can study the serving architecture before installing GPU-heavy dependencies.

## Files

| File | Chapter | Focus | Runs without ML dependencies |
| --- | --- | --- | --- |
| `ch2_llm_serving_examples.py` | Chapter 2 | LLM inference internals: tokenizer, model config, attention, manual decode, KV cache, vLLM basics | Only `setup` and `vllm-config` |
| `ch2_results.md` | Chapter 2 | Recorded command outputs and explanations | Yes |
| `ch3_model_serving_system_design.py` | Chapter 3 | Single-model serving, batching, streaming, multi-model LRU, Triton/vLLM wrappers | Yes, with `--backend mock` |
| `ch4_model_serving_best_practices_hw.py` | Chapter 4 | Knowledge Agent, RAG vs CAG, enterprise API, routing, metrics, build-vs-cloud decisions | Yes |
| `ch6_vllm_optimization_techniques_hw.py` | Chapter 6 | vLLM optimization techniques: batching, chunked prefill, attention, PagedAttention, quantization, prefix caching | Yes |
| `ch6_results.md` | Chapter 6 | Recorded comparison results and explanations | Yes |
| `ch9_llm_optimization_practice_hw.py` | Chapter 9 | Optimization lab: hardware inspection, synthetic traffic, vLLM commands, benchmark simulation, quantization, distributed serving trade-offs | Yes |

## Quick Start

From the repository root:

```bash
cd examples
python3 -B ch2_llm_serving_examples.py
python3 -B ch3_model_serving_system_design.py --section basic
python3 -B ch4_model_serving_best_practices_hw.py --section agent
python3 -B ch6_vllm_optimization_techniques_hw.py --section batching
python3 -B ch9_llm_optimization_practice_hw.py --section benchmark
```

The `-B` flag prevents Python from creating `__pycache__` files while you are
experimenting.

## Optional Dependencies

Base Python is enough for the mock/local sections. Install optional packages only
when you want to run the real model or web-service variants.

```bash
pip install fastapi uvicorn
pip install torch transformers accelerate matplotlib
pip install bertviz
pip install vllm
pip install tritonclient[http] requests numpy
```

Notes:

- `torch`, `transformers`, and model downloads are needed for real Hugging Face model sections.
- `vllm` generally expects a Linux/CUDA environment.
- Chapter 9 can generate real `vllm serve` and `vllm bench serve` commands, but its default benchmark is synthetic and local.

## Chapter 2: LLM Serving Internals

Show setup instructions:

```bash
python3 -B ch2_llm_serving_examples.py
```

Run sections that require Hugging Face dependencies:

```bash
python3 -B ch2_llm_serving_examples.py --section tokenizer
python3 -B ch2_llm_serving_examples.py --section config
python3 -B ch2_llm_serving_examples.py --section decoder
python3 -B ch2_llm_serving_examples.py --section attention
python3 -B ch2_llm_serving_examples.py --section pipeline
python3 -B ch2_llm_serving_examples.py --section manual-no-cache --max-new-tokens 20
python3 -B ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 20
python3 -B ch2_llm_serving_examples.py --section compare-manual --max-new-tokens 20
```

Run vLLM-related examples:

```bash
python3 -B ch2_llm_serving_examples.py --section vllm-config
python3 -B ch2_llm_serving_examples.py --section vllm-basic
python3 -B ch2_llm_serving_examples.py --section vllm-batch
python3 -B ch2_llm_serving_examples.py --section vllm-stream
```

### Chapter 2 Section Map

| Section | Chapter concept | Requires | Expected output shape |
| --- | --- | --- | --- |
| `setup` | How to run the chapter examples | Base Python | Setup instructions and suggested commands |
| `tokenizer` | Tokenizer: text -> tokens -> token IDs | `torch`, `transformers` | Token count plus token/id rows |
| `config` | Decoder-only model architecture and sizing | `torch`, `transformers` | Hidden size, layers, heads, vocab size, parameter count |
| `decoder` | Transformer decoder block structure | `torch`, `transformers` | Module tree with attention/MLP/layernorm modules |
| `attention` | Attention weights and token relationships | `torch`, `transformers`; optional `bertviz` | BertViz view or textual top-attention summary |
| `pipeline` | Hugging Face generation pipeline | `torch`, `transformers` | Completed text for the prompt |
| `manual-no-cache` | Autoregressive decoding without KV cache | `torch`, `transformers` | Token-by-token output with increasing per-token latency |
| `manual-kv-cache` | Prefill/decode split and KV-cache reuse | `torch`, `transformers` | First step as `prefill`, later steps as `decode` |
| `compare-manual` | Latency comparison: no cache vs KV cache | `torch`, `transformers`; optional `matplotlib` | Two generation runs and optional latency chart |
| `vllm-config` | vLLM serving knobs | Base Python | Printed vLLM config snippet |
| `vllm-basic` | vLLM single prompt generation | `vllm` | Generated text and elapsed time |
| `vllm-batch` | Batch serving with vLLM | `vllm` | Batch time vs one-by-one time |
| `vllm-stream` | Async streaming with vLLM | `vllm` | Incremental streamed text |

Sample output:

```text
$ python3 -B ch2_llm_serving_examples.py
Chapter 2 LLM serving examples
Install the Hugging Face examples:
    pip install torch transformers accelerate matplotlib

$ python3 -B ch2_llm_serving_examples.py --section vllm-config
from vllm import LLM, SamplingParams
model = LLM(
    model="Qwen/Qwen2.5-7B",
    ...
)
```

## Chapter 3: Serving System Design

Run local mock demos:

```bash
python3 -B ch3_model_serving_system_design.py --section basic
python3 -B ch3_model_serving_system_design.py --section batch
python3 -B ch3_model_serving_system_design.py --section stream
python3 -B ch3_model_serving_system_design.py --section multimodel
python3 -B ch3_model_serving_system_design.py --section triton
```

Run with a real Hugging Face backend:

```bash
python3 -B ch3_model_serving_system_design.py --section basic --backend transformers
python3 -B ch3_model_serving_system_design.py --section batch --backend transformers
```

Start FastAPI demos:

```bash
python3 -B ch3_model_serving_system_design.py --serve --port 8000
python3 -B ch3_model_serving_system_design.py --multi-serve --port 8001
```

### Chapter 3 Section Map

| Section | Chapter concept | Requires | Expected output shape |
| --- | --- | --- | --- |
| `basic` | Single generation request through engine/executor/worker | Base Python with `--backend mock` | One generated completion |
| `batch` | FIFO batching and response mapping | Base Python with `--backend mock` | Multiple prompts returned in input order |
| `stream` | Streaming with batching and per-request queues | Base Python with `--backend mock` | SSE-style `data: {"token": ...}` events |
| `multimodel` | Multi-model serving, model metadata, LRU eviction | Base Python | Cache state and prediction output per model |
| `vllm` | Replacing manual batching with vLLM | `vllm` | vLLM generated text per prompt |
| `triton` | Triton management/inference API wrapper concept | Base Python | `curl` examples and wrapper explanation |
| `--serve` | FastAPI single-model serving API | `fastapi`, `uvicorn` | HTTP API on `/basic_generate`, `/generate`, `/generate_stream` |
| `--multi-serve` | FastAPI multi-model API | `fastapi`, `uvicorn` | HTTP API on `/predict` |

Sample output:

```text
$ python3 -B ch3_model_serving_system_design.py --section basic
Hello, I am a compact model serving demo.

$ python3 -B ch3_model_serving_system_design.py --section stream
data: {"token": " a", "sequence_id": "..."}
data: {"token": " compact", "sequence_id": "..."}
...
```

## Chapter 4: Best Practices

Run local homework sections:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section agent
python3 -B ch4_model_serving_best_practices_hw.py --section rag --chunk-words 25
python3 -B ch4_model_serving_best_practices_hw.py --section cag
python3 -B ch4_model_serving_best_practices_hw.py --section routing
python3 -B ch4_model_serving_best_practices_hw.py --section enterprise
python3 -B ch4_model_serving_best_practices_hw.py --section metrics
python3 -B ch4_model_serving_best_practices_hw.py --section metrics --burst
python3 -B ch4_model_serving_best_practices_hw.py --section decision
```

Start the enterprise-style FastAPI API:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --serve --port 8000
```

### Chapter 4 Section Map

| Section | Chapter concept | Requires | Expected output shape |
| --- | --- | --- | --- |
| `agent` | Knowledge Agent: planner, actions, RAG, final synthesis | Base Python | JSON with plan, observations, final answer |
| `rag` | Retrieval-augmented generation and chunk retrieval | Base Python | Retrieved chunks, scores, context size |
| `cag` | Cache-augmented generation with preloaded context | Base Python | Context size and response without retrieval step |
| `routing` | Enterprise model selection, canary, tenant override | Base Python | Endpoint choices, strategy, HPA/Nginx YAML |
| `enterprise` | Auth, tenant identity, rate limit, route selection | Base Python | Routed mock chat completion JSON |
| `metrics` | E2E latency, TTFT, TPOT/ITL, RPS/RPM/TPS | Base Python | Summary metrics and example traces |
| `decision` | Build-versus-cloud decision framework | Base Python | Recommendations for prototype/custom/hybrid scenarios |
| `--serve` | Enterprise-style FastAPI API | `fastapi`, `uvicorn` | HTTP API on `/health` and `/v1/chat/completions` |

Sample output:

```json
$ python3 -B ch4_model_serving_best_practices_hw.py --section enterprise
{
  "tenant": "enterprise",
  "model": "gpt-4o-mini",
  "endpoint": "https://enterprise.example/v1/gpt-4o-mini",
  "strategy": "speculative_decode",
  "response": "This is a routed mock completion."
}
```

## Chapter 6: Essential LLM Optimization Techniques

Run local vLLM optimization homework sections:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section commands
python3 -B ch6_vllm_optimization_techniques_hw.py --section experiment-plan
python3 -B ch6_vllm_optimization_techniques_hw.py --section batching
python3 -B ch6_vllm_optimization_techniques_hw.py --section chunked-prefill
python3 -B ch6_vllm_optimization_techniques_hw.py --section attention
python3 -B ch6_vllm_optimization_techniques_hw.py --section paged-attention
python3 -B ch6_vllm_optimization_techniques_hw.py --section quantization
python3 -B ch6_vllm_optimization_techniques_hw.py --section prefix-cache
python3 -B ch6_vllm_optimization_techniques_hw.py --section all
```

### Chapter 6 Section Map

| Section | Chapter concept | Requires | Expected output shape |
| --- | --- | --- | --- |
| `commands` | vLLM flags for batching, chunked prefill, prefix caching, attention backend, AWQ/GPTQ/FP8 | Base Python | Copyable `vllm serve` and `vllm bench serve` commands |
| `experiment-plan` | Chapter 6 optimization experiment matrix | Base Python | Experiment-to-concept mapping, synthetic sweeps, and copyable vLLM commands |
| `batching` | Dynamic batching vs continuous batching | Base Python | TPS, TTFT, ITL, E2E comparison |
| `chunked-prefill` | Continuous batching with chunked prefill | Base Python | Comparison plus TTFT/ITL trade-off explanation |
| `attention` | MHA vs GQA vs MQA KV-cache memory | Base Python | KV-cache GiB and memory reduction ratios |
| `paged-attention` | PagedAttention and KV-cache fragmentation | Base Python | Contiguous vs paged token waste percentage |
| `quantization` | Model compression through lower precision | Base Python | FP32/FP16/INT8/INT4 memory estimate |
| `prefix-cache` | Prefix caching, prompt formatting, tenant isolation, cache-aware routing | Base Python | Prefix hit demo and serving metrics comparison |
| `all` | Run all local Chapter 6 examples | Base Python | All section outputs in sequence |

### Chapter 6 Experiment Plan

| 實驗 | 對應 Chapter 6 技術 |
| --- | --- |
| 調 `max-num-seqs` | continuous batching |
| 調 `max-num-batched-tokens` | token-level batching |
| 開 / 關 chunked prefill | chunked prefill |
| 原始模型 vs GPTQ / AWQ | quantization |
| 原始模型 vs FP8 | W8A8 quantization |
| default attention vs FlashInfer | attention kernel optimization |

Use `experiment-plan` first when designing a real vLLM benchmark run:

```bash
python3 -B ch6_vllm_optimization_techniques_hw.py --section experiment-plan --num-requests 40
```

Sample output:

```json
$ python3 -B ch6_vllm_optimization_techniques_hw.py --section attention
{
  "mha_gib": 16.0,
  "gqa_8kv_heads_gib": 4.0,
  "mqa_1kv_head_gib": 0.5,
  "gqa_memory_reduction_vs_mha": 4.0,
  "mqa_memory_reduction_vs_mha": 32.0
}
```

## Chapter 9: Optimization in Practice

Run local optimization lab sections:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section hardware
python3 -B ch9_llm_optimization_practice_hw.py --section dataset --dataset sharegpt
python3 -B ch9_llm_optimization_practice_hw.py --section dataset --dataset prefix
python3 -B ch9_llm_optimization_practice_hw.py --section commands
python3 -B ch9_llm_optimization_practice_hw.py --section parse-log
python3 -B ch9_llm_optimization_practice_hw.py --section benchmark
python3 -B ch9_llm_optimization_practice_hw.py --section quantization
python3 -B ch9_llm_optimization_practice_hw.py --section tuning
python3 -B ch9_llm_optimization_practice_hw.py --section distributed
python3 -B ch9_llm_optimization_practice_hw.py --section recommend --dataset prefix --latency-priority
```

Run all Chapter 9 sections:

```bash
python3 -B ch9_llm_optimization_practice_hw.py --section all
```

### Chapter 9 Section Map

| Section | Chapter concept | Requires | Expected output shape |
| --- | --- | --- | --- |
| `hardware` | Step 1: inspect GPU hardware and serving readiness | Base Python; optional `nvidia-smi` | GPU info or fallback sample L40S data |
| `dataset` | Step 2: inspect ShareGPT-like or prefix-repetition traffic | Base Python | Prompt/output length stats, histogram, sample prompts |
| `commands` | Step 4/5: generate vLLM serve and benchmark commands | Base Python | Copyable `vllm serve` and `vllm bench serve` commands |
| `parse-log` | Step 4: parse vLLM memory and KV-cache logs | Base Python | Model memory, KV cache memory, KV cache tokens |
| `benchmark` | Step 5/6: compare baseline, prefix traffic, AWQ | Base Python | TPS, TTFT, ITL for synthetic benchmark runs |
| `quantization` | Step 6: AWQ memory and throughput analysis | Base Python | Memory saved, KV-cache gain, TPS gain |
| `tuning` | Step 7: tune batching/cache knobs | Base Python | Top configs by total TPS plus homework prompts |
| `distributed` | Step 8: tensor parallelism and GPU interconnect trade-offs | Base Python | L40S/PCIe vs A100/NVLink analysis |
| `recommend` | Choosing optimization techniques for a traffic profile | Base Python | Optimization recommendations |
| `all` | Run the full local Chapter 9 lab | Base Python | All section outputs in sequence |

Sample output:

```json
$ python3 -B ch9_llm_optimization_practice_hw.py --section quantization
{
  "baseline": {
    "model_memory_gib": 27.5185,
    "kv_cache_memory_gib": 11.0,
    "kv_cache_tokens": 72064
  },
  "awq": {
    "model_memory_gib": 9.3619,
    "kv_cache_memory_gib": 29.15,
    "kv_cache_tokens": 191056
  },
  "kv_cache_token_gain": 2.6511989342806395,
  "chapter_sharegpt_total_tps_gain": 2.70042194092827
}
```

## Suggested Study Order

1. Chapter 2 `setup`, then `tokenizer`, `config`, and `manual-kv-cache`.
2. Chapter 3 `basic`, `batch`, `stream`, and `multimodel`.
3. Chapter 4 `agent`, `rag`, `cag`, `routing`, and `metrics --burst`.
4. Chapter 6 `commands`, `batching`, `attention`, `quantization`, and `prefix-cache`.
5. Chapter 9 `dataset`, `benchmark`, `quantization`, `tuning`, and `distributed`.

## Troubleshooting

If a section says dependencies are missing, either install the package listed in
the error message or run one of the mock/local sections first.

If a model section is slow on macOS or CPU, reduce generation length:

```bash
python3 -B ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 5
```

If a vLLM section fails locally, use the Chapter 9 `commands` section to generate
the commands and run them on a Linux/CUDA machine.
