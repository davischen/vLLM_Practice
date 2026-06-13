# Chapter 3 Model Serving System Design Results and Explanations

This file records local mock outputs from
`ch3_model_serving_system_design.py` and explains which Chapter 3 system-design
concept each result demonstrates.

The default backend is `mock`, so these results run without downloading a real
model. Use `--backend transformers` only when you want to test the same serving
flow with Hugging Face model execution.

## Performance and Cost Summary

| Serving pattern | Observed result | Performance or cost lever | Operational lesson |
| --- | --- | --- | --- |
| Basic single request | One prompt returns one completed string | Lowest orchestration overhead | Best starting point for API shape and correctness tests. |
| Batch serving | Five prompts are served together and returned in order | Amortizes backend execution overhead | Higher throughput requires preserving prompt-to-output mapping. |
| Streaming | Three concurrent streams emit token events with `sequence_id` | Improves perceived latency and TTFT experience | The client can render partial output before the full completion finishes. |
| Multi-model cache | Cache holds two models and evicts the least recently used model | Controls model-memory cost | LRU policy trades reload latency for bounded memory usage. |
| Triton wrapper | App delegates load and infer operations to Triton APIs | Reduces custom runtime ownership | Useful when model execution should be managed by a specialized serving backend. |

## Result 1: Basic Single-request Serving

Command:

```bash
python3 -B ch3_model_serving_system_design.py --section basic
```

Key output:

```text
Hello, I am a compact model serving demo.
```

Chapter concept:

Single-model serving path: request input enters the serving layer, the backend
generates a completion, and the response is returned as one completed string.

Explanation:

The mock backend appends deterministic tokens to the prompt. This is useful for
testing the serving architecture, API shape, and response formatting before
paying the cost of a real model.

## Result 2: Batch Serving

Command:

```bash
python3 -B ch3_model_serving_system_design.py --section batch
```

Key output:

```text
Prompt: Hello, I am
Output: Hello, I am a compact model serving demo.

Prompt: The weather is
Output: The weather is a compact model serving demo.

Prompt: I want to
Output: I want to a compact model serving demo.

Prompt: The best way to
Output: The best way to a compact model serving demo.

Prompt: The most efficient way to
Output: The most efficient way to a compact model serving demo.
```

Chapter concept:

Batch serving groups several prompts into one backend call while preserving the
mapping from each input prompt to its output.

Explanation:

The important behavior is ordering and response association. Even when multiple
requests are processed together, the serving layer must return the correct
completion to the correct caller.

## Result 3: Streaming Responses

Command:

```bash
python3 -B ch3_model_serving_system_design.py --section stream
```

Key output:

```text
stream start: Hello, I am

stream start: I want to

stream start: The best way to
data: {"token": " a", "sequence_id": "90f86a37-6dca-4c7a-a3ff-628552736f9a"}
data: {"token": " a", "sequence_id": "d2f6acc9-a43b-4f3e-bee0-5c7900723057"}
data: {"token": " a", "sequence_id": "c00027ad-a982-4e21-9da7-0f421b45f251"}
data: {"token": " compact", "sequence_id": "90f86a37-6dca-4c7a-a3ff-628552736f9a"}
data: {"token": " compact", "sequence_id": "d2f6acc9-a43b-4f3e-bee0-5c7900723057"}
data: {"token": " compact", "sequence_id": "c00027ad-a982-4e21-9da7-0f421b45f251"}
data: {"token": " model", "sequence_id": "90f86a37-6dca-4c7a-a3ff-628552736f9a"}
data: {"token": " model", "sequence_id": "d2f6acc9-a43b-4f3e-bee0-5c7900723057"}
data: {"token": " model", "sequence_id": "c00027ad-a982-4e21-9da7-0f421b45f251"}
data: {"token": " serving", "sequence_id": "90f86a37-6dca-4c7a-a3ff-628552736f9a"}
data: {"token": " serving", "sequence_id": "d2f6acc9-a43b-4f3e-bee0-5c7900723057"}
data: {"token": " serving", "sequence_id": "c00027ad-a982-4e21-9da7-0f421b45f251"}
data: {"token": " demo", "sequence_id": "90f86a37-6dca-4c7a-a3ff-628552736f9a"}
data: {"token": " demo", "sequence_id": "d2f6acc9-a43b-4f3e-bee0-5c7900723057"}
data: {"token": " demo", "sequence_id": "c00027ad-a982-4e21-9da7-0f421b45f251"}
data: {"token": ".", "sequence_id": "90f86a37-6dca-4c7a-a3ff-628552736f9a"}
stream end: Hello, I am
data: {"token": ".", "sequence_id": "d2f6acc9-a43b-4f3e-bee0-5c7900723057"}
stream end: I want to
data: {"token": ".", "sequence_id": "c00027ad-a982-4e21-9da7-0f421b45f251"}
stream end: The best way to
```

Chapter concept:

Streaming serving returns tokens incrementally instead of waiting for the full
completion. The service needs per-request queues or request IDs so each token
can be routed back to the right client.

Explanation:

The `data: ...` lines imitate Server-Sent Events style streaming. Three prompts
are active at the same time, and each generated token carries a different
`sequence_id`. That is why the same token, such as `" a"` or `" compact"`, is
emitted three times: one token event for each active request.

## Result 4: Multi-model Serving and LRU Eviction

Command:

```bash
python3 -B ch3_model_serving_system_design.py --section multimodel
```

Key output:

```text
Cache after loading 550e8400-e29b-41d4-a716-446655440000: ['550e8400-e29b-41d4-a716-446655440000']
{'model_id': '550e8400-e29b-41d4-a716-446655440000', 'label': 'POSITIVE', 'predictions': [[0.25, 0.75]]}

Cache after loading 7c9e6679-7425-40de-944b-e07fc1f90ae7: ['550e8400-e29b-41d4-a716-446655440000', '7c9e6679-7425-40de-944b-e07fc1f90ae7']
{'model_id': '7c9e6679-7425-40de-944b-e07fc1f90ae7', 'top_class': 'demo_object', 'predictions': [{'label': 'demo_object', 'score': 0.88}], 'input_summary': "{'image': 'pretend image bytes'}"}

Cache after loading 550e8400-e29b-41d4-a716-446655440000: ['7c9e6679-7425-40de-944b-e07fc1f90ae7', '550e8400-e29b-41d4-a716-446655440000']
{'model_id': '550e8400-e29b-41d4-a716-446655440000', 'label': 'POSITIVE', 'predictions': [[0.45, 0.55]]}
Evicted least recently used model: 7c9e6679-7425-40de-944b-e07fc1f90ae7

Cache after loading 11111111-2222-3333-4444-555555555555: ['550e8400-e29b-41d4-a716-446655440000', '11111111-2222-3333-4444-555555555555']
{'model_id': '11111111-2222-3333-4444-555555555555', 'prediction': 2.5}
```

Chapter concept:

Multi-model serving keeps model metadata and loaded model instances in a cache.
When memory is limited, the service evicts the least recently used model.

Explanation:

The cache initially loads a text classification model, then an image
classification model. When the text model is accessed again, it becomes recently
used and moves to the end of the LRU order. Loading the regression model then
evicts the image model because it is now the least recently used entry.

## Result 5: Triton Wrapper Concept

Command:

```bash
python3 -B ch3_model_serving_system_design.py --section triton
```

Key output:

```text
# Load a model into Triton:
curl -X POST http://localhost:8000/v2/repository/models/densenet_onnx/load

# Run inference through Triton:
curl -X POST http://localhost:8000/v2/models/densenet_onnx/infer

# In this file, TritonWorker wraps those management/inference APIs so a
# multi-model service can keep its own model metadata, cache policy, and public
# /predict interface while delegating model execution to Triton.
```

Chapter concept:

Triton can manage model loading and inference execution while your service keeps
the public API, metadata, cache policy, and routing logic.

Explanation:

This section does not require a Triton server. It prints the management and
inference API shape so you can see how a custom service would delegate model
execution to Triton while still owning the application-facing `/predict`
interface.
