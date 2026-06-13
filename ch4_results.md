# Chapter 4 Model Serving Best Practices Results and Explanations

This file records local outputs from
`ch4_model_serving_best_practices_hw.py` and maps them to Chapter 4 best
practices: agents, RAG/CAG, routing, enterprise controls, metrics, and
build-versus-cloud choices.

All results here use local deterministic mocks, so they run without ML
dependencies. The `--serve` result uses FastAPI and Uvicorn.

## Performance and Cost Summary

| Scenario | Key result | Performance or cost lesson |
| --- | --- | --- |
| RAG | Retrieval time: 0.0000505s, context: 589 chars | Request-time retrieval adds lookup work but keeps prompt context focused. |
| CAG | Retrieval time: 0.0000005s, context: 1,194 chars | Cached context can reduce retrieval latency, but may increase prompt-token cost. |
| Routing | Short request uses passthrough; long request uses speculative decode | Route by request shape so expensive strategies are used only when useful. |
| Normal metrics | RPS 1.3139, TPS 120.4858, p95 E2E 18.3454s, p95 TTFT 17.5272s | Track throughput and latency together; one number is not enough. |
| Burst metrics | RPS 1.3563, TPS 121.7617, p95 E2E 21.6412s, p95 TTFT 21.2691s | Similar throughput can still hide worse tail latency under bursty traffic. |
| Build vs cloud | Managed API, hybrid, and self-hosting are compared qualitatively | Cloud lowers ops cost; self-hosting raises control and infrastructure responsibility. |

## Result 1: Knowledge Agent

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section agent
```

Key output:

```json
{
  "query": "Create a detailed comparison between database query optimization and data structure optimization.",
  "plan": {
    "plan": [
      "query_rag_with_context",
      "generate_analysis",
      "generate_summary"
    ],
    "reasoning": "Retrieve grounding context first, then synthesize the answer.",
    "estimated_steps": 3
  },
  "observations": [
    {
      "action": "query_rag_with_context",
      "output": "Document 1 (source=database_queries.txt, score=0.338): database query optimization improves how relational systems execute sql ..."
    },
    {
      "action": "generate_analysis",
      "output": "Analysis: The retrieved context shows a trade-off between precision, latency, and system complexity. Query optimization and data-structure optimization both reduce wasted work, but operate at different layers."
    },
    {
      "action": "generate_summary",
      "output": "Summary: Ground the answer in retrieved or cached knowledge, then keep the final response concise and operationally useful."
    }
  ],
  "final_answer": "Summary: Ground the answer in retrieved or cached knowledge, then keep the final response concise and operationally useful.",
  "elapsed_seconds": 4.3e-05
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
  "retrieval_seconds": 5.054101347923279e-05,
  "context_chars": 589,
  "response": "Summary: Ground the answer in retrieved or cached knowledge, then keep the final response concise and operationally useful.",
  "context": "Document 1 (source=database_queries.txt, score=0.374): database query optimization improves how relational systems execute sql common query types include selection projection joins aggregation and subqueries optimizers compare candidate plans estimate costs\nDocument 2 (source=llm_serving.txt, score=0.276): single user request can trigger many model calls and tool calls\nDocument 3 (source=llm_serving.txt, score=0.194): kv cache reuse observability rate limits tenant isolation and cost control agentic workloads amplify latency because a single user request can trigger many model calls"
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
  "retrieval_seconds": 5.00120222568512e-07,
  "context_chars": 1194,
  "response": "Summary: Ground the answer in retrieved or cached knowledge, then keep the final response concise and operationally useful."
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
  "hpa_yaml": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\nmetadata: { name: enterprise-model-api-hpa }\nspec:\n  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: enterprise-model-api }\n  minReplicas: 3\n  maxReplicas: 15\n  metrics:\n  - type: Resource\n    resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }",
  "nginx_rate_limit_yaml": "apiVersion: networking.k8s.io/v1\nkind: Ingress\nmetadata:\n  name: api-ingress\n  annotations:\n    kubernetes.io/ingress.class: nginx\n    nginx.ingress.kubernetes.io/limit-rps: \"50\"\n    nginx.ingress.kubernetes.io/limit-burst-multiplier: \"5\"\n    nginx.ingress.kubernetes.io/proxy-body-size: \"8m\"\nspec:\n  rules:\n  - host: api.yourorg.example\n    http:\n      paths:\n      - path: /\n        pathType: Prefix\n        backend: { service: { name: enterprise-model-api, port: { number: 80 } } }"
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
  "burst": false,
  "summary": {
    "requests": 40.0,
    "duration_seconds": 30.443416017998324,
    "rps": 1.313912997685666,
    "rpm": 78.83477986113996,
    "tps": 120.48582188777557,
    "p50_e2e": 9.556604958348334,
    "p95_e2e": 18.345443578943538,
    "p50_ttft": 8.68020497194732,
    "p95_ttft": 17.527188562761204,
    "p50_tpot": 0.008162743542128568
  },
  "example_traces": [
    {
      "request_id": "req-0",
      "e2e_latency": 0.5543,
      "ttft": 0.0639,
      "tpot": 0.0086,
      "output_tokens": 58
    }
  ]
}
```

Chapter concept:

LLM serving should track end-to-end latency, time to first token, time per output
token, throughput, and tail latency.

Explanation:

This result separates user-perceived latency (`e2e`, `ttft`) from decode speed
(`tpot`) and service throughput (`rps`, `rpm`, `tps`). Those metrics answer
different operational questions.

## Result 7: Metrics With Bursty Traffic

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section metrics --burst
```

Key output:

```json
{
  "burst": true,
  "summary": {
    "requests": 40.0,
    "duration_seconds": 29.492022678475422,
    "rps": 1.3562989706092206,
    "rpm": 81.37793823655323,
    "tps": 121.76174008644277,
    "p50_e2e": 12.554364912429214,
    "p95_e2e": 21.641169875123403,
    "p50_ttft": 11.677964926028201,
    "p95_ttft": 21.269148825659446,
    "p50_tpot": 0.007672491287140893
  }
}
```

Chapter concept:

Bursty arrivals increase queueing pressure. Even if throughput remains similar,
latency percentiles can get worse.

Explanation:

Compared with Result 6, bursty traffic raises p50 E2E latency from about
`9.56s` to `12.55s` and p95 E2E latency from about `18.35s` to `21.64s`.
This demonstrates why average throughput alone is not enough for serving SLOs.

## Result 8: Build-versus-cloud Decision

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --section decision
```

Key output:

```json
{
  "prototype": {
    "recommendation": "Option 1: fully managed foundation-model API",
    "reason": "Lowest operational overhead; good for quick prototypes and stable low-volume apps."
  },
  "custom_agent_platform": {
    "recommendation": "Option 5 or 6: bring your own image or serving stack",
    "reason": "You need control over batching, routing, kernels, autoscaling, telemetry, or portability."
  },
  "managed_with_custom_handler": {
    "recommendation": "Option 4: bring your own code in a managed container",
    "reason": "Custom request handling matters, but the vendor runtime is still acceptable."
  }
}
```

Chapter concept:

Serving architecture should match business constraints: speed of iteration,
control, portability, customization, compliance, and operations burden.

Explanation:

There is no universal best deployment style. The result frames the decision as a
trade-off between managed simplicity and custom control.

## Result 9: FastAPI Enterprise API Root Endpoint

Command:

```bash
python3 -B ch4_model_serving_best_practices_hw.py --serve --port 8000
```

Server output:

```text
INFO:     Started server process [99761]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:49368 - "GET / HTTP/1.1" 200 OK
```

Browser or `curl http://127.0.0.1:8000/` result:

```json
{
  "name": "Chapter 4 enterprise model API homework",
  "status": "ok",
  "endpoints": {
    "health": "GET /health",
    "docs": "GET /docs",
    "chat_completions": "POST /v1/chat/completions"
  },
  "auth": {
    "api_key_header": "x-api-key",
    "demo_api_key": "demo-key",
    "bearer_header": "Authorization: Bearer demo-token"
  }
}
```

Chapter concept:

Production APIs should expose discoverable health and documentation endpoints.
The model endpoint itself still requires authentication and a structured request
body.

Explanation:

The first server run returned `404 Not Found` for `GET /` because only `/health`
and `/v1/chat/completions` were defined. After adding the root endpoint, `GET /`
returns a small API index with available routes and demo authentication headers.
Use `/docs` for the generated FastAPI UI, `/health` for health checks, and
`POST /v1/chat/completions` for the enterprise chat demo.
