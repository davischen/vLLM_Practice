"""Chapter 8 LLM serving frameworks homework.

This local lab turns Chapter 8's serving-framework ideas into runnable examples.
It focuses on vLLM's architecture and Scheduler, then compares vLLM with
TensorRT-LLM, SGLang, and llama.cpp.

The default sections are synthetic and run on any machine. They do not require a
GPU or vLLM. Use the command-builder section when you want copyable commands for
a real Linux/CUDA vLLM environment.

Covered concepts:
    - Why LLM serving frameworks differ from traditional model servers.
    - vLLM LLM class and OpenAI-compatible API server usage.
    - LLMEngine, EngineCore, Scheduler, ModelExecutor, GPUWorker, ModelRunner.
    - Multi-process worker initialization.
    - Request execution workflow from Processor to OutputProcessor.
    - vLLM-style WAITING/RUNNING queues and token-level scheduling.
    - Token budget, max-num-seqs, chunked prefill, prefix caching, preemption.
    - Framework-selection trade-offs across vLLM, TensorRT-LLM, SGLang,
      and llama.cpp.

Examples:
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section commands
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section architecture
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section init
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section workflow
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section scheduler
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section compare-schedulers
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section frameworks
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section evaluate
    python3 -B examples/ch8_llm_serving_frameworks_hw.py --section all
"""

# %%
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


DEFAULT_MODEL = "Qwen/Qwen3-7B-Instruct"


# %%
# Shared helpers


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


def round_dict(data: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    rounded = {}
    for key, value in data.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        else:
            rounded[key] = value
    return rounded


# %%
# 1. vLLM command and API examples


def vllm_library_snippet(model: str = DEFAULT_MODEL) -> str:
    return f"""
from vllm import LLM, SamplingParams

llm = LLM(
    model="{model}",
    trust_remote_code=True,
    dtype="float16",
    max_model_len=32768,
    gpu_memory_utilization=0.8,
)

sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
outputs = llm.generate([
    "Hello, my name is",
    "Explain vLLM scheduling in one paragraph.",
], sampling)

for output in outputs:
    print(output.prompt, output.outputs[0].text)
""".strip()


def vllm_server_command(model: str = DEFAULT_MODEL) -> str:
    return f"""
vllm serve {model} \\
  --trust-remote-code \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --gpu-memory-utilization 0.8 \\
  --max-num-seqs 128 \\
  --max-num-batched-tokens 8192 \\
  --enable-chunked-prefill \\
  --enable-prefix-caching
""".strip()


def openai_compatible_curl(model: str = DEFAULT_MODEL) -> str:
    return f"""
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model}",
    "messages": [
      {{"role": "user", "content": "Explain continuous batching."}}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
  }}'
""".strip()


def run_commands(model: str = DEFAULT_MODEL) -> None:
    print("=== vLLM LLM class ===")
    print(vllm_library_snippet(model))
    print("\n=== vLLM OpenAI-compatible server ===")
    print(vllm_server_command(model))
    print("\n=== OpenAI-compatible client call ===")
    print(openai_compatible_curl(model))


# %%
# 2. Architecture and initialization


def architecture_map() -> dict[str, Any]:
    return {
        "chapter_concept": "vLLM separates public API, scheduling, execution, worker lifecycle, and model running.",
        "layers": [
            {
                "component": "LLMEngine",
                "responsibility": "Public entry point; receives generation requests and coordinates lifecycle.",
            },
            {
                "component": "EngineCore",
                "responsibility": "Inner loop; asks Scheduler for work and passes outputs to processors.",
            },
            {
                "component": "Scheduler",
                "responsibility": "Maintains WAITING/RUNNING queues, token budgets, KV blocks, and scheduling policy.",
            },
            {
                "component": "ModelExecutor",
                "responsibility": "Orchestrates worker processes and dispatches SchedulerOutput for execution.",
            },
            {
                "component": "GPUWorker",
                "responsibility": "Owns device state, communication, and model lifecycle in each worker process.",
            },
            {
                "component": "GPUModelRunner",
                "responsibility": "Builds the actual model inputs and runs forward passes.",
            },
        ],
        "optimization_layers": {
            "Scheduler": "System-wide, model-agnostic optimizations such as continuous batching, prefix caching, and chunked prefill.",
            "ModelExecutor": "Architecture-specific execution and distributed worker orchestration.",
            "Model layers": "Attention, MLP, KV-cache reuse, and fused operations.",
            "CustomOp": "Hardware-specific kernels such as CUDA or tensor-core optimized operators.",
        },
    }


def run_architecture() -> None:
    print_json(architecture_map())


def simulate_initialization(tensor_parallel_size: int = 4, backend: str = "mp") -> dict[str, Any]:
    events = [
        {
            "step": 1,
            "actor": "main process",
            "event": "Create LLM config and initialize LLMEngine, EngineCore, Scheduler, KVCacheManager, and executor.",
        },
        {
            "step": 2,
            "actor": "MultiProcessExecutor",
            "event": f"Spawn {tensor_parallel_size} worker processes with distributed_executor_backend={backend!r}.",
        },
        {
            "step": 3,
            "actor": "GPUWorker",
            "event": "Each worker sets CUDA device, initializes communication, and prepares response queues.",
        },
        {
            "step": 4,
            "actor": "GPUModelRunner",
            "event": f"Look up model implementation for {DEFAULT_MODEL} and load sharded weights.",
        },
    ]
    workers = [
        {
            "rank": rank,
            "tensor_parallel_size": tensor_parallel_size,
            "executor_backend": backend,
            "queue": f"worker_response_mq_rank_{rank}",
            "model_shard": f"tp_shard_{rank + 1}_of_{tensor_parallel_size}",
        }
        for rank in range(tensor_parallel_size)
    ]
    return {
        "llm_constructor": {
            "model": DEFAULT_MODEL,
            "tensor_parallel_size": tensor_parallel_size,
            "distributed_executor_backend": backend,
        },
        "events": events,
        "workers": workers,
        "chapter_concept": "vLLM initializes framework components in the main process, then creates worker processes for model execution.",
    }


def run_initialization(tensor_parallel_size: int = 4) -> None:
    print_json(simulate_initialization(tensor_parallel_size=tensor_parallel_size))


def simulate_generation_workflow() -> dict[str, Any]:
    return {
        "input": {
            "prompts": ["Hello, my name is", "Explain vLLM scheduling."],
            "sampling_params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 64},
        },
        "workflow": [
            {
                "stage": "Processor",
                "output": "Validate inputs, tokenize prompts, create Request objects.",
            },
            {
                "stage": "LLMEngine / EngineCore",
                "output": "Enter execution loop and ask Scheduler to build the next batch.",
            },
            {
                "stage": "Scheduler",
                "output": {
                    "scheduled_requests": ["req-1", "req-2"],
                    "num_scheduled_tokens": {"req-1": 18, "req-2": 24},
                    "metadata": ["KV block assignment", "attention metadata", "sampling metadata"],
                },
            },
            {
                "stage": "ModelExecutor",
                "output": "Broadcast SchedulerOutput to workers and run model forward pass.",
            },
            {
                "stage": "OutputProcessor",
                "output": "Convert token IDs into streamed chunks or final RequestOutput objects.",
            },
        ],
        "chapter_concept": "SchedulerOutput is the work order that connects scheduling decisions to model execution.",
    }


def run_workflow() -> None:
    print_json(simulate_generation_workflow())


# %%
# 3. vLLM-style Scheduler simulation


@dataclass
class SimRequest:
    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    priority: int = 0
    prefix_id: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass
class RequestState:
    request: SimRequest
    num_computed_tokens: int = 0
    first_token_ms: float | None = None
    completion_ms: float | None = None
    preemptions: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.request.total_tokens - self.num_computed_tokens)

    @property
    def in_prefill(self) -> bool:
        return self.num_computed_tokens < self.request.prompt_tokens

    @property
    def output_computed(self) -> int:
        return max(0, self.num_computed_tokens - self.request.prompt_tokens)


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 512
    long_prefill_token_threshold: int = 128
    enable_prefix_cache: bool = True
    preemption_mode: str = "recompute"
    policy: str = "priority"


@dataclass
class ScheduleStep:
    step: int
    now_ms: float
    token_budget: int
    scheduled_tokens: dict[str, int]
    running: list[str]
    waiting: list[str]
    preempted: list[str] = field(default_factory=list)
    prefix_cache_hits: list[str] = field(default_factory=list)
    phase: dict[str, str] = field(default_factory=dict)
    step_ms: float = 0.0


@dataclass
class SchedulerSummary:
    strategy: str
    requests: int
    completed: int
    duration_ms: float
    total_prompt_tokens: int
    total_output_tokens: int
    total_scheduled_tokens: int
    request_throughput_rps: float
    total_tps: float
    output_tps: float
    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_e2e_ms: float
    p95_e2e_ms: float
    mean_step_scheduled_tokens: float
    token_budget_utilization: float
    prefix_cache_hit_rate: float
    preemptions: int


def generate_scheduler_requests(count: int = 12, seed: int = 8, traffic: str = "mixed") -> list[SimRequest]:
    rng = random.Random(seed)
    now = 0.0
    rows: list[SimRequest] = []
    for idx in range(count):
        now += rng.expovariate(7.0) * 1000.0
        if traffic == "long_prefill":
            prompt_tokens = rng.randint(900, 2200)
            output_tokens = rng.randint(32, 160)
        elif traffic == "decode_heavy":
            prompt_tokens = rng.randint(64, 260)
            output_tokens = rng.randint(250, 700)
        else:
            prompt_tokens = rng.choice([96, 160, 384, 768, 1400])
            output_tokens = rng.randint(64, 280)
        priority = 2 if idx in {3, 7} else rng.choice([0, 0, 1])
        prefix_id = idx % 3 if idx % 2 == 0 else None
        rows.append(
            SimRequest(
                request_id=f"req-{idx + 1}",
                arrival_ms=now,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                priority=priority,
                prefix_id=prefix_id,
            )
        )
    return rows


def sort_waiting(waiting: deque[RequestState], policy: str) -> deque[RequestState]:
    if policy == "priority":
        ordered = sorted(waiting, key=lambda state: (-state.request.priority, state.request.arrival_ms))
        return deque(ordered)
    return waiting


def maybe_apply_prefix_cache(
    state: RequestState,
    prefix_cache: set[int],
    config: SchedulerConfig,
) -> bool:
    prefix_id = state.request.prefix_id
    if not config.enable_prefix_cache or prefix_id is None or state.num_computed_tokens != 0:
        return False
    if prefix_id in prefix_cache:
        cached_tokens = int(state.request.prompt_tokens * 0.7)
        state.num_computed_tokens = max(state.num_computed_tokens, cached_tokens)
        return True
    prefix_cache.add(prefix_id)
    return False


def maybe_preempt(
    running: list[RequestState],
    waiting: deque[RequestState],
    config: SchedulerConfig,
) -> list[str]:
    if not waiting or not running or config.preemption_mode == "none":
        return []
    top_waiting = waiting[0]
    lowest_running = min(running, key=lambda state: (state.request.priority, -state.request.arrival_ms))
    if top_waiting.request.priority <= lowest_running.request.priority:
        return []

    running.remove(lowest_running)
    lowest_running.preemptions += 1
    if config.preemption_mode == "recompute":
        lowest_running.num_computed_tokens = 0
        lowest_running.first_token_ms = None
    waiting.append(lowest_running)
    return [lowest_running.request.request_id]


def estimate_step_ms(prefill_tokens: int, decode_tokens: int, scheduled_count: int) -> float:
    base = 1.2
    prefill_ms = prefill_tokens * 0.022
    decode_ms = 0.0
    if decode_tokens:
        decode_ms = 5.0 / math.sqrt(max(1, decode_tokens))
    overhead_ms = scheduled_count * 0.08
    return base + prefill_ms + decode_ms + overhead_ms


def simulate_vllm_scheduler(
    requests: list[SimRequest],
    config: SchedulerConfig | None = None,
    max_steps: int = 5000,
) -> tuple[SchedulerSummary, list[ScheduleStep]]:
    if config is None:
        config = SchedulerConfig()

    queued = deque(sorted([RequestState(req) for req in requests], key=lambda state: state.request.arrival_ms))
    waiting: deque[RequestState] = deque()
    running: list[RequestState] = []
    prefix_cache: set[int] = set()
    completed: list[RequestState] = []
    steps: list[ScheduleStep] = []
    now = min(req.arrival_ms for req in requests) if requests else 0.0
    prefix_hits = 0
    prefix_lookups = 0
    total_token_budget = 0
    total_scheduled_tokens = 0
    preemptions = 0

    for step_idx in range(max_steps):
        while queued and queued[0].request.arrival_ms <= now:
            waiting.append(queued.popleft())
        waiting = sort_waiting(waiting, config.policy)

        if len(running) >= config.max_num_seqs:
            preempted = maybe_preempt(running, waiting, config)
            preemptions += len(preempted)
        else:
            preempted = []

        waiting = sort_waiting(waiting, config.policy)
        while waiting and len(running) < config.max_num_seqs:
            running.append(waiting.popleft())

        if not running:
            if queued:
                now = queued[0].request.arrival_ms
                continue
            break

        token_budget = config.max_num_batched_tokens
        scheduled: dict[str, int] = {}
        phase: dict[str, str] = {}
        prefix_hit_ids: list[str] = []
        prefill_tokens = 0
        decode_tokens = 0

        running.sort(key=lambda state: (-state.request.priority, state.request.arrival_ms))
        for state in list(running):
            if token_budget <= 0:
                break

            if state.request.prefix_id is not None and state.num_computed_tokens == 0:
                prefix_lookups += 1
            if maybe_apply_prefix_cache(state, prefix_cache, config):
                prefix_hits += 1
                prefix_hit_ids.append(state.request.request_id)

            gap = state.remaining_tokens
            if gap <= 0:
                continue

            if state.in_prefill:
                prefill_left = state.request.prompt_tokens - state.num_computed_tokens
                if config.long_prefill_token_threshold > 0:
                    num_new_tokens = min(prefill_left, config.long_prefill_token_threshold)
                else:
                    num_new_tokens = prefill_left
                token_phase = "prefill"
            else:
                num_new_tokens = 1
                token_phase = "decode"

            num_new_tokens = min(num_new_tokens, token_budget, gap)
            if num_new_tokens <= 0:
                continue

            state.num_computed_tokens += num_new_tokens
            token_budget -= num_new_tokens
            scheduled[state.request.request_id] = num_new_tokens
            phase[state.request.request_id] = token_phase
            if token_phase == "prefill":
                prefill_tokens += num_new_tokens
            else:
                decode_tokens += num_new_tokens

        if not scheduled:
            now += 1.0
            continue

        step_ms = estimate_step_ms(prefill_tokens, decode_tokens, len(scheduled))
        step = ScheduleStep(
            step=step_idx + 1,
            now_ms=round(now, 4),
            token_budget=config.max_num_batched_tokens,
            scheduled_tokens=scheduled,
            running=[state.request.request_id for state in running],
            waiting=[state.request.request_id for state in waiting],
            preempted=preempted,
            prefix_cache_hits=prefix_hit_ids,
            phase=phase,
            step_ms=round(step_ms, 4),
        )
        steps.append(step)
        total_token_budget += config.max_num_batched_tokens
        total_scheduled_tokens += sum(scheduled.values())
        now += step_ms

        still_running = []
        for state in running:
            if state.num_computed_tokens > state.request.prompt_tokens and state.first_token_ms is None:
                state.first_token_ms = now
            if state.remaining_tokens <= 0:
                state.completion_ms = now
                completed.append(state)
            else:
                still_running.append(state)
        running = still_running

        if len(completed) == len(requests):
            break

    summary = summarize_scheduler(
        "vllm_token_level_scheduler",
        requests,
        completed,
        total_scheduled_tokens,
        total_token_budget,
        prefix_hits,
        prefix_lookups,
        preemptions,
        steps,
    )
    return summary, steps


def simulate_request_level_batcher(
    requests: list[SimRequest],
    max_batch_size: int = 4,
) -> SchedulerSummary:
    queued = deque(sorted(requests, key=lambda req: req.arrival_ms))
    now = min(req.arrival_ms for req in requests) if requests else 0.0
    completed = []
    total_scheduled = 0
    total_budget = 0
    while queued:
        if queued[0].arrival_ms > now:
            now = queued[0].arrival_ms
        batch = []
        while queued and len(batch) < max_batch_size and queued[0].arrival_ms <= now:
            batch.append(queued.popleft())
        if not batch and queued:
            batch.append(queued.popleft())

        batch_tokens = sum(req.total_tokens for req in batch)
        batch_prefill = sum(req.prompt_tokens for req in batch)
        batch_decode = sum(req.output_tokens for req in batch)
        step_ms = 8.0 + batch_prefill * 0.030 + batch_decode * 0.24 / math.sqrt(max(1, len(batch)))
        start_ms = now
        now += step_ms
        total_scheduled += batch_tokens
        total_budget += max(batch_tokens, 1)
        for req in batch:
            state = RequestState(req)
            state.num_computed_tokens = req.total_tokens
            state.first_token_ms = start_ms + 8.0 + req.prompt_tokens * 0.030
            state.completion_ms = now
            completed.append(state)

    return summarize_scheduler(
        "request_level_batcher",
        requests,
        completed,
        total_scheduled,
        total_budget,
        0,
        0,
        0,
        [],
    )


def summarize_scheduler(
    strategy: str,
    requests: list[SimRequest],
    completed: list[RequestState],
    total_scheduled_tokens: int,
    total_token_budget: int,
    prefix_hits: int,
    prefix_lookups: int,
    preemptions: int,
    steps: list[ScheduleStep],
) -> SchedulerSummary:
    if not requests:
        return SchedulerSummary(strategy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    start = min(req.arrival_ms for req in requests)
    end = max((state.completion_ms or start) for state in completed) if completed else start
    duration_ms = max(1.0, end - start)
    ttfts = [
        (state.first_token_ms or state.completion_ms or end) - state.request.arrival_ms
        for state in completed
    ]
    e2es = [
        (state.completion_ms or end) - state.request.arrival_ms
        for state in completed
    ]
    total_prompt = sum(req.prompt_tokens for req in requests)
    total_output = sum(req.output_tokens for req in requests)
    step_tokens = [sum(step.scheduled_tokens.values()) for step in steps]
    return SchedulerSummary(
        strategy=strategy,
        requests=len(requests),
        completed=len(completed),
        duration_ms=duration_ms,
        total_prompt_tokens=total_prompt,
        total_output_tokens=total_output,
        total_scheduled_tokens=total_scheduled_tokens,
        request_throughput_rps=len(completed) / (duration_ms / 1000.0),
        total_tps=(total_prompt + total_output) / (duration_ms / 1000.0),
        output_tps=total_output / (duration_ms / 1000.0),
        mean_ttft_ms=statistics.mean(ttfts) if ttfts else 0.0,
        p95_ttft_ms=percentile(ttfts, 95),
        mean_e2e_ms=statistics.mean(e2es) if e2es else 0.0,
        p95_e2e_ms=percentile(e2es, 95),
        mean_step_scheduled_tokens=statistics.mean(step_tokens) if step_tokens else total_scheduled_tokens,
        token_budget_utilization=total_scheduled_tokens / total_token_budget if total_token_budget else 1.0,
        prefix_cache_hit_rate=prefix_hits / prefix_lookups if prefix_lookups else 0.0,
        preemptions=preemptions,
    )


def summary_brief(summary: SchedulerSummary) -> dict[str, Any]:
    return round_dict(asdict(summary), digits=4)


def step_brief(step: ScheduleStep) -> dict[str, Any]:
    return asdict(step)


def run_scheduler_demo() -> None:
    requests = generate_scheduler_requests(count=10, seed=8, traffic="mixed")
    config = SchedulerConfig(
        max_num_seqs=3,
        max_num_batched_tokens=384,
        long_prefill_token_threshold=128,
        enable_prefix_cache=True,
        preemption_mode="recompute",
        policy="priority",
    )
    summary, steps = simulate_vllm_scheduler(requests, config=config)
    print_json(
        {
            "config": asdict(config),
            "requests": [asdict(req) for req in requests],
            "summary": summary_brief(summary),
            "first_scheduler_outputs": [step_brief(step) for step in steps[:10]],
            "chapter_concept": "vLLM schedules tokens, not whole requests. RUNNING requests are handled first, while WAITING requests are admitted when sequence slots and token budget allow.",
        }
    )


def run_scheduler_comparison() -> None:
    requests = generate_scheduler_requests(count=40, seed=18, traffic="long_prefill")
    request_level = simulate_request_level_batcher(requests, max_batch_size=4)
    token_basic, _ = simulate_vllm_scheduler(
        requests,
        config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=1024,
            long_prefill_token_threshold=0,
            enable_prefix_cache=False,
            preemption_mode="none",
            policy="fcfs",
        ),
    )
    token_optimized, _ = simulate_vllm_scheduler(
        requests,
        config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            long_prefill_token_threshold=256,
            enable_prefix_cache=True,
            preemption_mode="swap",
            policy="priority",
        ),
    )
    rows = [summary_brief(item) for item in [request_level, token_basic, token_optimized]]
    print_json(
        {
            "traffic": "long_prefill",
            "comparison": rows,
            "performance_lesson": "Token-level scheduling, chunked prefill, and prefix caching reduce long-prompt head-of-line blocking and improve TTFT in this synthetic workload.",
            "cost_lesson": "Prefix cache lowers scheduled-token work; larger concurrency can improve fleet efficiency, but the final choice must still satisfy memory and latency SLOs.",
        }
    )


# %%
# 4. Serving framework comparison


def framework_matrix() -> list[dict[str, str]]:
    return [
        {
            "framework": "vLLM",
            "best_for": "Fast path to production, broad Hugging Face model support, OpenAI-compatible serving.",
            "strengths": "PagedAttention, continuous batching, prefix caching, speculative decoding, strong ecosystem.",
            "cost_profile": "Good datacenter GPU utilization with moderate operational complexity.",
        },
        {
            "framework": "TensorRT-LLM",
            "best_for": "NVIDIA-centric production stacks chasing peak tokens per dollar.",
            "strengths": "TensorRT engines, FP8/FP4/INT4, in-flight batching, Triton/Dynamo integration.",
            "cost_profile": "High performance on NVIDIA hardware, higher build and engine-management complexity.",
        },
        {
            "framework": "SGLang",
            "best_for": "Agentic, structured-output, multi-step, and multi-vendor workloads.",
            "strengths": "RadixAttention, structured generation, router, continuous batching, broad hardware targets.",
            "cost_profile": "Strong for complex serving flows; evaluate ecosystem and ops maturity for your team.",
        },
        {
            "framework": "llama.cpp",
            "best_for": "Local, private, on-device, edge, and low-cost inference.",
            "strengths": "GGUF quantization, tiny dependency footprint, CPU/Metal/CUDA/Vulkan portability.",
            "cost_profile": "Lowest ops and infrastructure footprint, but not aimed at high-concurrency datacenter TPS.",
        },
    ]


def run_frameworks() -> None:
    print_json(
        {
            "selection_rule": "Choose by SLOs, workload shape, hardware, operability, and portability.",
            "frameworks": framework_matrix(),
            "chapter_concept": "There is no permanent best serving framework; reassess as models, kernels, and hardware change.",
        }
    )


@dataclass
class WorkloadProfile:
    latency_priority: bool = True
    high_throughput: bool = True
    nvidia_only: bool = False
    structured_outputs: bool = False
    edge_or_private: bool = False
    broad_model_support: bool = True


def evaluate_frameworks(profile: WorkloadProfile) -> list[dict[str, Any]]:
    scores = {"vLLM": 0, "TensorRT-LLM": 0, "SGLang": 0, "llama.cpp": 0}
    reasons: dict[str, list[str]] = {name: [] for name in scores}

    if profile.latency_priority:
        for name in ["vLLM", "TensorRT-LLM", "SGLang"]:
            scores[name] += 2
            reasons[name].append("Designed for low-latency online serving.")
    if profile.high_throughput:
        scores["vLLM"] += 2
        scores["TensorRT-LLM"] += 3
        scores["SGLang"] += 2
        reasons["vLLM"].append("Strong continuous batching baseline.")
        reasons["TensorRT-LLM"].append("Excellent peak throughput on NVIDIA GPUs.")
        reasons["SGLang"].append("Competitive runtime with batching and KV reuse.")
    if profile.nvidia_only:
        scores["TensorRT-LLM"] += 4
        reasons["TensorRT-LLM"].append("Deep NVIDIA/TensorRT integration.")
    else:
        scores["vLLM"] += 1
        scores["SGLang"] += 1
        scores["llama.cpp"] += 1
        reasons["vLLM"].append("Portable enough across common open-model GPU setups.")
        reasons["SGLang"].append("Designed with multi-vendor serving in mind.")
        reasons["llama.cpp"].append("Portable across local CPU/GPU backends.")
    if profile.structured_outputs:
        scores["SGLang"] += 4
        reasons["SGLang"].append("Strong structured generation and agent workflow support.")
    if profile.edge_or_private:
        scores["llama.cpp"] += 5
        reasons["llama.cpp"].append("Low-footprint local and private serving.")
    if profile.broad_model_support:
        scores["vLLM"] += 3
        scores["SGLang"] += 1
        reasons["vLLM"].append("Large ecosystem and broad Hugging Face model coverage.")
        reasons["SGLang"].append("Broad open-model support with strong backend features.")

    return [
        {
            "framework": name,
            "score": score,
            "reasons": reasons[name],
        }
        for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def run_evaluate(args: argparse.Namespace) -> None:
    profile = WorkloadProfile(
        latency_priority=args.latency_priority,
        high_throughput=args.high_throughput,
        nvidia_only=args.nvidia_only,
        structured_outputs=args.structured_outputs,
        edge_or_private=args.edge_or_private,
        broad_model_support=not args.narrow_model_support,
    )
    print_json(
        {
            "profile": asdict(profile),
            "ranking": evaluate_frameworks(profile),
            "chapter_concept": "Framework choice should start from SLOs, workload shape, hardware reality, and operational constraints.",
        }
    )


# %%
# Command-line entrypoint


SECTIONS = {
    "commands": run_commands,
    "architecture": run_architecture,
    "init": run_initialization,
    "workflow": run_workflow,
    "scheduler": run_scheduler_demo,
    "compare-schedulers": run_scheduler_comparison,
    "frameworks": run_frameworks,
}


def run_all(args: argparse.Namespace) -> None:
    run_commands(args.model)
    print("\n=== Architecture ===")
    run_architecture()
    print("\n=== Initialization ===")
    run_initialization(args.tensor_parallel_size)
    print("\n=== Workflow ===")
    run_workflow()
    print("\n=== Scheduler ===")
    run_scheduler_demo()
    print("\n=== Scheduler comparison ===")
    run_scheduler_comparison()
    print("\n=== Frameworks ===")
    run_frameworks()
    print("\n=== Evaluate ===")
    run_evaluate(args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        default="scheduler",
        choices=sorted([*SECTIONS.keys(), "evaluate", "all"]),
        help="Which Chapter 8 example to run",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID for generated vLLM commands")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--latency-priority", action="store_true", default=True)
    parser.add_argument("--no-latency-priority", dest="latency_priority", action="store_false")
    parser.add_argument("--high-throughput", action="store_true", default=True)
    parser.add_argument("--low-throughput", dest="high_throughput", action="store_false")
    parser.add_argument("--nvidia-only", action="store_true")
    parser.add_argument("--structured-outputs", action="store_true")
    parser.add_argument("--edge-or-private", action="store_true")
    parser.add_argument("--narrow-model-support", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.section == "commands":
        run_commands(args.model)
    elif args.section == "init":
        run_initialization(args.tensor_parallel_size)
    elif args.section == "evaluate":
        run_evaluate(args)
    elif args.section == "all":
        run_all(args)
    else:
        SECTIONS[args.section]()


if __name__ == "__main__":
    main()
