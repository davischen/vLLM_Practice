# Chapter 4 Model Serving Best Practices Results and Explanations

This file records local outputs from
`ch4_model_serving_best_practices_hw.py` and maps them to Chapter 4 best
practices: agents, RAG/CAG, routing, enterprise controls, metrics, and
build-versus-cloud choices.

All results here use local deterministic mocks, so they run without ML
dependencies.

## Result 1: Knowledge Agent

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section agent
```

Key output:

```json
{
  "plan": ["query_rag_with_context", "generate_analysis", "generate_summary"],
  "reasoning": "Retrieve grounding context first, then synthesize the answer.",
  "observations": [
    {"action": "query_rag_with_context", "output": "Document 1 ..."},
    {"action": "generate_analysis", "output": "Analysis: ..."},
    {"action": "generate_summary", "output": "Summary: ..."}
  ],
  "final_answer": "Summary: Ground the answer in retrieved or cached knowledge..."
}
```

Chapter concept:

A knowledge agent decomposes a user request into tool or retrieval steps, records
observations, and synthesizes a final answer.

Explanation:

The output shows the agent loop explicitly: plan, retrieve, analyze, summarize.
This is useful when serving systems need traceability rather than just a final
text response.

## Result 2: RAG

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section rag --chunk-words 25
```

Key output:

```json
{
  "mode": "RAG",
  "retrieval_seconds": 0.000049,
  "context_chars": 589,
  "response": "Summary: Ground the answer in retrieved or cached knowledge...",
  "context": "Document 1 (source=database_queries.txt, score=0.374): ..."
}
```

Chapter concept:

Retrieval-augmented generation retrieves a small set of relevant chunks at
request time and passes them as context to the model.

Explanation:

The result includes retrieval time, context size, source file names, and scores.
These are the serving-time signals you would log when diagnosing retrieval
quality or latency.

## Result 3: CAG

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section cag
```

Key output:

```json
{
  "mode": "CAG",
  "retrieval_seconds": 0.00000058,
  "context_chars": 1194,
  "response": "Summary: Ground the answer in retrieved or cached knowledge..."
}
```

Chapter concept:

Cache-augmented generation preloads or caches context so a request can avoid a
fresh retrieval step.

Explanation:

Compared with RAG, CAG trades freshness and flexibility for lower retrieval
latency. This is helpful when the same corpus or prompt prefix is reused often.

## Result 4: Routing and Operations Configuration

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section routing
```

Key output:

```json
{
  "dev_default": "https://models.example/v1/gpt-4o-mini",
  "dev_canary": "https://canary.example/v1/gpt-4o-mini",
  "enterprise_override": "https://enterprise.example/v1/gpt-4o-mini",
  "short_strategy": "passthrough",
  "long_strategy": "speculative_decode",
  "hpa_yaml": "apiVersion: autoscaling/v2...",
  "nginx_rate_limit_yaml": "apiVersion: networking.k8s.io/v1..."
}
```

Chapter concept:

Production serving often needs tenant-specific routing, canary endpoints,
request-length strategies, autoscaling, and rate limiting.

Explanation:

The output combines application-level routing decisions with infrastructure
configuration examples. This mirrors real serving systems, where reliability is
part code and part deployment policy.

## Result 5: Enterprise Chat Request

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section enterprise
```

Key output:

```json
{
  "tenant": "enterprise",
  "model": "gpt-4o-mini",
  "endpoint": "https://enterprise.example/v1/gpt-4o-mini",
  "strategy": "speculative_decode",
  "response": "This is a routed mock completion."
}
```

Chapter concept:

Enterprise serving adds tenant identity, rate limits, routing decisions, and
strategy selection around the raw model call.

Explanation:

The model response is only one field. The rest of the result shows operational
metadata that production systems need for auditing, billing, routing, and SLOs.

## Result 6: Metrics

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section metrics
```

Key output:

```json
{
  "requests": 40.0,
  "rps": 1.3139,
  "rpm": 78.8348,
  "tps": 120.4858,
  "p50_e2e": 9.5566,
  "p95_e2e": 18.3454,
  "p50_ttft": 8.6802,
  "p95_ttft": 17.5272,
  "p50_tpot": 0.00816
}
```

Chapter concept:

LLM serving should track end-to-end latency, time to first token, time per output
token, throughput, and tail latency.

Explanation:

This result separates user-perceived latency (`e2e`, `ttft`) from decode speed
(`tpot`) and service throughput (`rps`, `rpm`, `tps`). Those metrics answer
different operational questions.

## Result 7: Build-versus-cloud Decision

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section decision
```

Key output:

```json
{
  "prototype": {
    "recommendation": "Option 1: fully managed foundation-model API"
  },
  "custom_agent_platform": {
    "recommendation": "Option 5 or 6: bring your own image or serving stack"
  },
  "managed_with_custom_handler": {
    "recommendation": "Option 4: bring your own code in a managed container"
  }
}
```

Chapter concept:

Serving architecture should match business constraints: speed of iteration,
control, portability, customization, compliance, and operations burden.

Explanation:

There is no universal best deployment style. The result frames the decision as a
trade-off between managed simplicity and custom control.
