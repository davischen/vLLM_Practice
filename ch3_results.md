# Chapter 3 Model Serving System Design Results and Explanations

This file records local mock outputs from
`ch3_model_serving_system_design.py` and explains which Chapter 3 system-design
concept each result demonstrates.

The default backend is `mock`, so these results run without downloading a real
model. Use `--backend transformers` only when you want to test the same serving
flow with Hugging Face model execution.

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
python3 -B ch3_model_serving_system_design.py --section stream --max-tokens 6
```

Key output:

```text
stream start: Hello, I am
stream start: I want to
stream start: The best way to
data: {"token": " a", "sequence_id": "..."}
data: {"token": " compact", "sequence_id": "..."}
data: {"token": " model", "sequence_id": "..."}
data: {"token": " serving", "sequence_id": "..."}
data: {"token": " demo", "sequence_id": "..."}
data: {"token": ".", "sequence_id": "..."}
stream end: Hello, I am
```

Chapter concept:

Streaming serving returns tokens incrementally instead of waiting for the full
completion. The service needs per-request queues or request IDs so each token
can be routed back to the right client.

Explanation:

The `data: ...` lines imitate Server-Sent Events style streaming. Multiple
requests can be active at the same time, so every emitted token carries a
`sequence_id`.

## Result 4: Multi-model Serving and LRU Eviction

Command:

```bash
python3 -B ch3_model_serving_system_design.py --section multimodel
```

Key output:

```text
Cache after loading 550e8400-e29b-41d4-a716-446655440000:
['550e8400-e29b-41d4-a716-446655440000']
{'model_id': '550e8400-e29b-41d4-a716-446655440000', 'label': 'POSITIVE', ...}

Cache after loading 7c9e6679-7425-40de-944b-e07fc1f90ae7:
['550e8400-e29b-41d4-a716-446655440000', '7c9e6679-7425-40de-944b-e07fc1f90ae7']
{'model_id': '7c9e6679-7425-40de-944b-e07fc1f90ae7', 'top_class': 'demo_object', ...}

Evicted least recently used model: 7c9e6679-7425-40de-944b-e07fc1f90ae7
```

Chapter concept:

Multi-model serving keeps model metadata and loaded model instances in a cache.
When memory is limited, the service evicts the least recently used model.

Explanation:

This result shows a practical model-management concern that does not appear in
single-model demos: the serving layer must decide what to load, what to keep,
and what to evict as traffic shifts across model IDs.

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
```

Chapter concept:

Triton can manage model loading and inference execution while your service keeps
the public API, metadata, cache policy, and routing logic.

Explanation:

This section does not require a Triton server. It prints the management and
inference API shape so you can see how a custom service would delegate model
execution to Triton.
