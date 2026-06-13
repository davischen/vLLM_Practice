"""Chapter 4 model serving best-practices homework.

This program implements runnable versions of the chapter's examples:

1. A minimal Knowledge Agent with planner, RAG, actions, and final synthesis.
2. RAG versus CAG behavior using an in-memory knowledge base.
3. A small enterprise-style API layer with auth, rate limits, and routing.
4. Model endpoint selection with tenant overrides and canary routing.
5. LLM serving metrics: E2E latency, TTFT, ITL/TPOT, RPS, RPM, and TPS.
6. Build-versus-cloud decision guidance as executable rules.

The default backend is deterministic and local. No OpenAI key, vector database,
Kubernetes cluster, or AWS account is required.

Optional installs:
    pip install fastapi uvicorn

Examples:
    python3 -B examples/ch4_model_serving_best_practices_hw.py --section agent
    python3 -B examples/ch4_model_serving_best_practices_hw.py --section rag
    python3 -B examples/ch4_model_serving_best_practices_hw.py --section cag
    python3 -B examples/ch4_model_serving_best_practices_hw.py --section metrics
    python3 -B examples/ch4_model_serving_best_practices_hw.py --section routing
    python3 -B examples/ch4_model_serving_best_practices_hw.py --section decision

Serve the FastAPI enterprise API demo:
    python3 -B examples/ch4_model_serving_best_practices_hw.py --serve --port 8000
"""

# %%
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# %%
# Dependency helpers


def optional_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_dependency(module_name: str, install_hint: str) -> None:
    if not optional_dependency(module_name):
        raise RuntimeError(f"{module_name!r} is not installed. Install it with `{install_hint}`.")


# %%
# Knowledge documents and lightweight embedding


DEFAULT_DOCS = {
    "database_queries.txt": """
Database query optimization improves how relational systems execute SQL.
Common query types include selection, projection, joins, aggregation, and
subqueries. Optimizers compare candidate plans, estimate costs, use indexes,
rewrite predicates, and choose join orders to reduce disk I/O and CPU time.
""",
    "data_structures.txt": """
Patricia tries are compressed prefix trees. They optimize lookup by collapsing
chains of single-child nodes, which saves memory and reduces traversal steps.
This makes them useful for routing tables, dictionaries, and string-keyed maps.
""",
    "llm_serving.txt": """
LLM serving systems care about end-to-end latency, time to first token, time per
output token, throughput, batching, KV cache reuse, observability, rate limits,
tenant isolation, and cost control. Agentic workloads amplify latency because a
single user request can trigger many model calls and tool calls.
""",
    "rag_cag.txt": """
Retrieval-augmented generation retrieves relevant chunks at query time and sends
them to the LLM as context. Cache-augmented generation preloads knowledge into
the model context or KV cache, reducing retrieval latency but increasing cache
and context management pressure.
""",
}


def tokenize(text: str) -> list[str]:
    return [
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}\"'")
    ]


def chunk_text(text: str, chunk_words: int = 45, overlap_words: int = 8) -> list[str]:
    """Split text into overlapping chunks.

    HOMEWORK: Try chunk_words=20 versus chunk_words=80 and compare retrieval
    precision in the --section rag demo.
    """

    words = tokenize(text)
    if not words:
        return []

    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def hashed_embedding(text: str, dims: int = 96) -> list[float]:
    """A tiny deterministic bag-of-words embedding for local demos."""

    vector = [0.0] * dims
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class DocumentChunk:
    source: str
    text: str
    embedding: list[float]


class InMemoryRAGSystem:
    """Offline index building plus online retrieval, all in memory."""

    def __init__(self, docs: dict[str, str], chunk_words: int = 45, overlap_words: int = 8):
        self.docs = docs
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.index: list[DocumentChunk] = []
        self.build_index()

    def build_index(self) -> None:
        self.index.clear()
        for source, text in self.docs.items():
            for chunk in chunk_text(text, self.chunk_words, self.overlap_words):
                self.index.append(
                    DocumentChunk(
                        source=source,
                        text=chunk,
                        embedding=hashed_embedding(chunk),
                    )
                )

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, DocumentChunk]]:
        query_embedding = hashed_embedding(query)
        scored = [
            (cosine_similarity(query_embedding, chunk.embedding), chunk)
            for chunk in self.index
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def context_for_query(self, query: str, top_k: int = 3) -> str:
        lines = []
        for rank, (score, chunk) in enumerate(self.search(query, top_k=top_k), start=1):
            lines.append(
                f"Document {rank} (source={chunk.source}, score={score:.3f}): {chunk.text}"
            )
        return "\n".join(lines)


class CacheAugmentedKnowledge:
    """A toy CAG cache that preloads all knowledge into a single context string."""

    def __init__(self, docs: dict[str, str]):
        self.preloaded_context = "\n\n".join(f"[{name}]\n{text.strip()}" for name, text in docs.items())
        self.loaded_at = time.time()

    def context_for_query(self, query: str) -> str:
        return self.preloaded_context


# %%
# Mock LLM manager, planner, and actions


class MockLLMManager:
    """Deterministic local stand-in for hosted LLM calls."""

    def create_planning_prompt(self, query: str, available_actions: list[str]) -> str:
        return (
            "Create a JSON plan for this user query.\n"
            f"Available actions: {available_actions}\n"
            f"Query: {query}"
        )

    def generate_response(self, prompt: str, temperature: float = 0.3) -> str:
        lower = prompt.lower()
        if "json plan" in lower:
            if any(word in lower for word in ["compare", "comparison", "relate", "analysis"]):
                plan = ["query_rag_with_context", "generate_analysis", "generate_summary"]
            elif "summar" in lower:
                plan = ["query_rag_with_context", "generate_summary"]
            else:
                plan = ["query_rag_with_context", "generate_summary"]
            return json.dumps(
                {
                    "plan": plan,
                    "reasoning": "Retrieve grounding context first, then synthesize the answer.",
                    "estimated_steps": len(plan),
                }
            )

        if "please provide a detailed analysis" in lower:
            return (
                "Analysis: The retrieved context shows a trade-off between precision, "
                "latency, and system complexity. Query optimization and data-structure "
                "optimization both reduce wasted work, but operate at different layers."
            )

        if "summarize" in lower or "summary" in lower:
            return (
                "Summary: Ground the answer in retrieved or cached knowledge, then keep "
                "the final response concise and operationally useful."
            )

        return (
            "Answer: Based on the available context, the system should retrieve relevant "
            "knowledge, reason over it, and respond with clear trade-offs."
        )


class Planner:
    def __init__(self, llm_manager: MockLLMManager, available_actions: list[str]):
        self.llm_manager = llm_manager
        self.available_actions = available_actions

    def create_plan(self, query: str) -> dict[str, Any]:
        planning_prompt = self.llm_manager.create_planning_prompt(query, self.available_actions)
        plan_response = self.llm_manager.generate_response(planning_prompt, temperature=0.3)
        return self._parse_plan_response(plan_response)

    def _parse_plan_response(self, plan_response: str) -> dict[str, Any]:
        try:
            parsed = json.loads(plan_response)
        except json.JSONDecodeError:
            parsed = {"plan": ["query_rag_with_context", "generate_summary"], "reasoning": "fallback"}
        parsed["plan"] = [
            action for action in parsed.get("plan", []) if action in self.available_actions
        ]
        if not parsed["plan"]:
            parsed["plan"] = ["query_rag_with_context", "generate_summary"]
        return parsed


class ActionExecutor:
    def __init__(self, llm_manager: MockLLMManager, rag_system: InMemoryRAGSystem):
        self.llm_manager = llm_manager
        self.rag_system = rag_system

    def create_analysis_prompt(self, query: str, context: str) -> str:
        return f"""
You are an expert analyst. Please provide a detailed analysis of the following
question based on the provided context.

Question: {query}

Context:
{context}

Please provide:
1. A comprehensive analysis
2. Key insights and findings
3. Relevant examples from the context
4. Any limitations or gaps in the available information

Analysis:
""".strip()

    def query_rag_with_context(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        context = self.rag_system.context_for_query(query, top_k=3)
        state["context"] = context
        state["observations"].append({"action": "query_rag_with_context", "output": context})
        return state

    def generate_analysis(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self.create_analysis_prompt(query, state.get("context", ""))
        analysis = self.llm_manager.generate_response(prompt, temperature=0.2)
        state["analysis"] = analysis
        state["observations"].append({"action": "generate_analysis", "output": analysis})
        return state

    def generate_summary(self, query: str, state: dict[str, Any]) -> dict[str, Any]:
        source = state.get("analysis") or state.get("context", "")
        prompt = f"Summarize the answer to '{query}' using this material:\n{source}"
        summary = self.llm_manager.generate_response(prompt, temperature=0.2)
        state["summary"] = summary
        state["observations"].append({"action": "generate_summary", "output": summary})
        return state

    def execute(self, action: str, query: str, state: dict[str, Any]) -> dict[str, Any]:
        return getattr(self, action)(query, state)


class KnowledgeAgent:
    def __init__(self, rag_system: Optional[InMemoryRAGSystem] = None):
        self.llm_manager = MockLLMManager()
        self.available_actions = [
            "query_rag_with_context",
            "generate_analysis",
            "generate_summary",
        ]
        self.rag_system = rag_system or InMemoryRAGSystem(DEFAULT_DOCS)
        self.planner = Planner(self.llm_manager, self.available_actions)
        self.actions = ActionExecutor(self.llm_manager, self.rag_system)

    def answer(self, query: str) -> dict[str, Any]:
        plan = self.planner.create_plan(query)
        state: dict[str, Any] = {"query": query, "observations": []}
        start = time.perf_counter()
        for action in plan["plan"]:
            state = self.actions.execute(action, query, state)
        state["final_answer"] = state.get("summary") or state.get("analysis") or ""
        state["plan"] = plan
        state["elapsed_seconds"] = time.perf_counter() - start
        return state


# %%
# RAG versus CAG comparison


def answer_with_rag(query: str, rag_system: InMemoryRAGSystem) -> dict[str, Any]:
    start = time.perf_counter()
    context = rag_system.context_for_query(query)
    retrieval_time = time.perf_counter() - start
    response = MockLLMManager().generate_response(
        f"Summarize the answer to '{query}' using this material:\n{context}"
    )
    return {
        "mode": "RAG",
        "retrieval_seconds": retrieval_time,
        "context_chars": len(context),
        "response": response,
        "context": context,
    }


def answer_with_cag(query: str, cag: CacheAugmentedKnowledge) -> dict[str, Any]:
    start = time.perf_counter()
    context = cag.context_for_query(query)
    retrieval_time = time.perf_counter() - start
    response = MockLLMManager().generate_response(
        f"Summarize the answer to '{query}' using this preloaded context:\n{context}"
    )
    return {
        "mode": "CAG",
        "retrieval_seconds": retrieval_time,
        "context_chars": len(context),
        "response": response,
    }


# %%
# Enterprise API concepts: auth, rate limit, endpoint selection


@dataclass
class ChatReq:
    model: str
    messages: list[dict[str, str]]
    max_new_tokens: int = 256


@dataclass
class TenantIdentity:
    tenant: str
    claims: dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None


class InMemoryIdentityProvider:
    def __init__(self):
        self.api_keys = {
            "dev-key": {"tenant": "dev"},
            "enterprise-key": {"tenant": "enterprise"},
        }

    async def require_auth(self, api_key: Optional[str] = None, bearer_token: Optional[str] = None) -> TenantIdentity:
        if bearer_token:
            try:
                claims = json.loads(bearer_token)
            except json.JSONDecodeError as exc:
                raise PermissionError("Bad JWT-like token") from exc
            tenant = claims.get("tenant")
            if tenant:
                return TenantIdentity(tenant=tenant, claims=claims)

        if api_key and api_key in self.api_keys:
            return TenantIdentity(tenant=self.api_keys[api_key]["tenant"], api_key=api_key)

        raise PermissionError("Missing or invalid API key/JWT")


class TokenBucketRateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.events: dict[str, deque[float]] = defaultdict(deque)

    async def rate_limit(self, tenant: str) -> None:
        now = time.time()
        window = 60.0
        bucket = self.events[tenant]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= self.requests_per_minute:
            raise RuntimeError(f"Rate limit exceeded for tenant={tenant}")
        bucket.append(now)


ROUTE_CONFIG = {
    "aliases": {
        "default": {
            "url": "https://models.example/v1/gpt-4o-mini",
            "cost_class": "low",
        }
    },
    "models": {
        "gpt-4o-mini": {
            "url": "https://models.example/v1/gpt-4o-mini",
            "draft_enabled": True,
            "canary": {"url": "https://canary.example/v1/gpt-4o-mini", "weight": 0.10},
            "tenants": {
                "enterprise": {"url": "https://enterprise.example/v1/gpt-4o-mini"}
            },
        },
        "gpt-4.1": {
            "url": "https://models.example/v1/gpt-4.1",
            "draft_enabled": False,
        },
    },
}


def choose_endpoint(model: str, tenant: str, random_value: Optional[float] = None) -> str:
    cfg = ROUTE_CONFIG
    route = cfg["models"].get(model) or cfg["aliases"].get(model)
    if not route:
        raise KeyError(f"Unknown model {model}")

    route_override = (route.get("tenants") or {}).get(tenant)
    if route_override:
        return route_override["url"]

    if "canary" in route:
        rand = random.random() if random_value is None else random_value
        if rand < float(route["canary"]["weight"]):
            return route["canary"]["url"]

    return route["url"]


def should_use_speculative_decode(req: ChatReq) -> bool:
    route = ROUTE_CONFIG["models"].get(req.model, {})
    return bool(route.get("draft_enabled")) and req.max_new_tokens > 1024


async def enterprise_chat(
    req: ChatReq,
    identity: TenantIdentity,
    limiter: TokenBucketRateLimiter,
) -> dict[str, Any]:
    await limiter.rate_limit(identity.tenant)
    endpoint = choose_endpoint(req.model, identity.tenant)
    strategy = "speculative_decode" if should_use_speculative_decode(req) else "passthrough"
    return {
        "tenant": identity.tenant,
        "model": req.model,
        "endpoint": endpoint,
        "strategy": strategy,
        "response": "This is a routed mock completion.",
    }


def hpa_yaml() -> str:
    return """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: enterprise-model-api-hpa }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: enterprise-model-api }
  minReplicas: 3
  maxReplicas: 15
  metrics:
  - type: Resource
    resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } }
""".strip()


def nginx_rate_limit_yaml() -> str:
    return """
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/limit-rps: "50"
    nginx.ingress.kubernetes.io/limit-burst-multiplier: "5"
    nginx.ingress.kubernetes.io/proxy-body-size: "8m"
spec:
  rules:
  - host: api.yourorg.example
    http:
      paths:
      - path: /
        pathType: Prefix
        backend: { service: { name: enterprise-model-api, port: { number: 80 } } }
""".strip()


def create_enterprise_app():
    require_dependency("fastapi", "pip install fastapi uvicorn")

    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="Chapter 4 enterprise model API homework")
    idp = InMemoryIdentityProvider()
    limiter = TokenBucketRateLimiter(requests_per_minute=20)

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequestModel(BaseModel):
        model: str = "gpt-4o-mini"
        messages: list[Message]
        max_new_tokens: int = 256

    @app.get("/")
    async def root():
        return {
            "name": "Chapter 4 enterprise model API homework",
            "status": "ok",
            "endpoints": {
                "health": "GET /health",
                "docs": "GET /docs",
                "chat_completions": "POST /v1/chat/completions",
            },
            "auth": {
                "api_key_header": "x-api-key",
                "demo_api_key": "demo-key",
                "bearer_header": "Authorization: Bearer demo-token",
            },
        }

    async def require_auth(
        x_api_key: Optional[str] = Header(default=None),
        authorization: Optional[str] = Header(default=None),
    ) -> TenantIdentity:
        bearer_token = None
        if authorization:
            parts = authorization.split(" ", 1)
            if len(parts) != 2 or parts[0].lower() != "bearer":
                raise HTTPException(401, "Bad auth type")
            bearer_token = parts[1]
        try:
            return await idp.require_auth(api_key=x_api_key, bearer_token=bearer_token)
        except PermissionError as exc:
            raise HTTPException(401, str(exc)) from exc

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequestModel, identity: TenantIdentity = Depends(require_auth)):
        try:
            result = await enterprise_chat(
                ChatReq(
                    model=req.model,
                    messages=[message.model_dump() for message in req.messages],
                    max_new_tokens=req.max_new_tokens,
                ),
                identity,
                limiter,
            )
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return result

    return app


# %%
# Metrics: latency and throughput


@dataclass
class RequestTrace:
    request_id: str
    arrival_time: float
    first_token_time: float
    completion_time: float
    output_tokens: int

    @property
    def e2e_latency(self) -> float:
        return self.completion_time - self.arrival_time

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return (self.completion_time - self.first_token_time) / (self.output_tokens - 1)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


def summarize_traces(traces: list[RequestTrace]) -> dict[str, float]:
    duration = max(trace.completion_time for trace in traces) - min(trace.arrival_time for trace in traces)
    duration = max(duration, 1e-9)
    total_output_tokens = sum(trace.output_tokens for trace in traces)
    return {
        "requests": float(len(traces)),
        "duration_seconds": duration,
        "rps": len(traces) / duration,
        "rpm": 60 * len(traces) / duration,
        "tps": total_output_tokens / duration,
        "p50_e2e": percentile([trace.e2e_latency for trace in traces], 50),
        "p95_e2e": percentile([trace.e2e_latency for trace in traces], 95),
        "p50_ttft": percentile([trace.ttft for trace in traces], 50),
        "p95_ttft": percentile([trace.ttft for trace in traces], 95),
        "p50_tpot": percentile([trace.tpot for trace in traces], 50),
    }


def simulate_llm_traces(
    requests: int = 40,
    burst: bool = False,
    seed: int = 7,
) -> list[RequestTrace]:
    """Generate synthetic traces for benchmarking homework.

    HOMEWORK: Change burst=True and observe how queueing affects p95 E2E.
    """

    rng = random.Random(seed)
    traces = []
    now = 0.0
    server_available_at = 0.0
    for i in range(requests):
        interarrival = rng.expovariate(4.0) if not burst else rng.choice([0.0, 0.02, 0.05, 0.5])
        now += interarrival
        output_tokens = rng.randint(20, 160)
        prompt_tokens = rng.randint(20, 700)
        queue_start = max(now, server_available_at)
        prefill = 0.010 + prompt_tokens * 0.00008
        decode_first = 0.020
        tpot = 0.006 + rng.random() * 0.004
        first_token_time = queue_start + prefill + decode_first
        completion_time = first_token_time + max(0, output_tokens - 1) * tpot
        server_available_at = completion_time
        traces.append(
            RequestTrace(
                request_id=f"req-{i}",
                arrival_time=now,
                first_token_time=first_token_time,
                completion_time=completion_time,
                output_tokens=output_tokens,
            )
        )
    return traces


# %%
# Build-versus-cloud decision framework


@dataclass
class ServingRequirements:
    custom_model: bool
    custom_preprocessing: bool
    strict_latency_slo: bool
    high_traffic: bool
    compliance_or_lock_in_sensitive: bool
    small_team: bool


def recommend_serving_option(req: ServingRequirements) -> dict[str, str]:
    if req.small_team and not req.custom_model and not req.custom_preprocessing:
        option = "Option 1: fully managed foundation-model API"
        reason = "Lowest operational overhead; good for quick prototypes and stable low-volume apps."
    elif not req.custom_preprocessing and not req.strict_latency_slo:
        option = "Option 2 or 3: JumpStart / prebuilt serving container"
        reason = "You need more control than a serverless API but can reuse managed hosting."
    elif req.custom_preprocessing and not req.compliance_or_lock_in_sensitive:
        option = "Option 4: bring your own code in a managed container"
        reason = "Custom request handling matters, but the vendor runtime is still acceptable."
    elif req.strict_latency_slo or req.high_traffic or req.compliance_or_lock_in_sensitive:
        option = "Option 5 or 6: bring your own image or serving stack"
        reason = "You need control over batching, routing, kernels, autoscaling, telemetry, or portability."
    else:
        option = "Hybrid"
        reason = "Keep a stable API and customize only the high-value endpoints."
    return {"recommendation": option, "reason": reason}


# %%
# CLI demos


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def demo_agent(query: str) -> None:
    agent = KnowledgeAgent()
    result = agent.answer(query)
    print_json(
        {
            "query": query,
            "plan": result["plan"],
            "observations": result["observations"],
            "final_answer": result["final_answer"],
            "elapsed_seconds": round(result["elapsed_seconds"], 6),
        }
    )


def demo_rag(query: str, chunk_words: int) -> None:
    rag = InMemoryRAGSystem(DEFAULT_DOCS, chunk_words=chunk_words)
    print_json(answer_with_rag(query, rag))


def demo_cag(query: str) -> None:
    cag = CacheAugmentedKnowledge(DEFAULT_DOCS)
    print_json(answer_with_cag(query, cag))


def demo_routing() -> None:
    req_short = ChatReq(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello"}], max_new_tokens=256)
    req_long = ChatReq(model="gpt-4o-mini", messages=[{"role": "user", "content": "Write a book"}], max_new_tokens=2048)
    results = {
        "dev_default": choose_endpoint("gpt-4o-mini", "dev", random_value=0.5),
        "dev_canary": choose_endpoint("gpt-4o-mini", "dev", random_value=0.01),
        "enterprise_override": choose_endpoint("gpt-4o-mini", "enterprise", random_value=0.01),
        "short_strategy": "speculative_decode" if should_use_speculative_decode(req_short) else "passthrough",
        "long_strategy": "speculative_decode" if should_use_speculative_decode(req_long) else "passthrough",
        "hpa_yaml": hpa_yaml(),
        "nginx_rate_limit_yaml": nginx_rate_limit_yaml(),
    }
    print_json(results)


async def demo_enterprise_chat() -> None:
    idp = InMemoryIdentityProvider()
    limiter = TokenBucketRateLimiter(requests_per_minute=3)
    identity = await idp.require_auth(api_key="enterprise-key")
    req = ChatReq(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Explain serving metrics."}],
        max_new_tokens=1500,
    )
    print_json(await enterprise_chat(req, identity, limiter))


def demo_metrics(burst: bool) -> None:
    traces = simulate_llm_traces(requests=40, burst=burst)
    summary = summarize_traces(traces)
    examples = [
        {
            "request_id": trace.request_id,
            "e2e_latency": round(trace.e2e_latency, 4),
            "ttft": round(trace.ttft, 4),
            "tpot": round(trace.tpot, 4),
            "output_tokens": trace.output_tokens,
        }
        for trace in traces[:5]
    ]
    print_json({"burst": burst, "summary": summary, "example_traces": examples})


def demo_decision() -> None:
    scenarios = {
        "prototype": ServingRequirements(
            custom_model=False,
            custom_preprocessing=False,
            strict_latency_slo=False,
            high_traffic=False,
            compliance_or_lock_in_sensitive=False,
            small_team=True,
        ),
        "custom_agent_platform": ServingRequirements(
            custom_model=True,
            custom_preprocessing=True,
            strict_latency_slo=True,
            high_traffic=True,
            compliance_or_lock_in_sensitive=True,
            small_team=False,
        ),
        "managed_with_custom_handler": ServingRequirements(
            custom_model=True,
            custom_preprocessing=True,
            strict_latency_slo=False,
            high_traffic=False,
            compliance_or_lock_in_sensitive=False,
            small_team=False,
        ),
    }
    print_json({name: recommend_serving_option(req) for name, req in scenarios.items()})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["agent", "rag", "cag", "routing", "enterprise", "metrics", "decision"],
        default="agent",
    )
    parser.add_argument(
        "--query",
        default="Create a detailed comparison between database query optimization and data structure optimization.",
    )
    parser.add_argument("--chunk-words", type=int, default=45)
    parser.add_argument("--burst", action="store_true", help="Use bursty traffic for metrics simulation")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI enterprise API demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def serve_app(host: str, port: int) -> None:
    require_dependency("uvicorn", "pip install fastapi uvicorn")
    import uvicorn

    uvicorn.run(create_enterprise_app(), host=host, port=port)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.serve:
        serve_app(args.host, args.port)
        return

    if args.section == "agent":
        demo_agent(args.query)
    elif args.section == "rag":
        demo_rag(args.query, args.chunk_words)
    elif args.section == "cag":
        demo_cag(args.query)
    elif args.section == "routing":
        demo_routing()
    elif args.section == "enterprise":
        asyncio.run(demo_enterprise_chat())
    elif args.section == "metrics":
        demo_metrics(args.burst)
    elif args.section == "decision":
        demo_decision()


if __name__ == "__main__":
    main()
