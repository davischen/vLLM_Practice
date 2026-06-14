"""Local vLLM serving observability demo.

This script is a lightweight stand-in for a vLLM OpenAI-compatible server. It
exposes Prometheus-compatible vLLM-style metrics on /metrics and writes JSON
request logs for Loki/Promtail.

Use this when you want to test the Prometheus/Grafana/Loki stack without a GPU
or a real vLLM install. For a real server, run `vllm serve ...` and let
Prometheus scrape that server's /metrics endpoint.

Examples:
    python3 -B vllm_observability_demo.py
    python3 -B vllm_observability_demo.py --port 9108 --model Qwen/Qwen3-14B
    python3 -B vllm_observability_demo.py --no-simulated-traffic
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


HISTOGRAM_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


@dataclass
class Histogram:
    buckets: list[float]
    values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.values.append(value)

    def render(self, metric: str, labels: str) -> list[str]:
        lines = []
        for bucket in self.buckets:
            count = sum(1 for value in self.values if value <= bucket)
            lines.append(f'{metric}_bucket{{{labels},le="{bucket}"}} {count}')
        lines.append(f'{metric}_bucket{{{labels},le="+Inf"}} {len(self.values)}')
        lines.append(f"{metric}_count{{{labels}}} {len(self.values)}")
        lines.append(f"{metric}_sum{{{labels}}} {sum(self.values):.6f}")
        return lines


@dataclass
class ServingState:
    model: str
    running_requests: int = 0
    waiting_requests: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    successful_requests_total: int = 0
    failed_requests_total: int = 0
    gpu_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float = 0.0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttft_seconds: Histogram = field(default_factory=lambda: Histogram(HISTOGRAM_BUCKETS))
    e2e_seconds: Histogram = field(default_factory=lambda: Histogram(HISTOGRAM_BUCKETS))


class ObservableVllmServer:
    def __init__(
        self,
        *,
        model: str,
        log_path: Path,
        seed: int,
        interval_seconds: float,
    ) -> None:
        self.state = ServingState(model=model)
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.logger = self._build_logger(log_path)

    def _build_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("vllm_observability_demo")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)
        return logger

    def metrics_text(self) -> str:
        with self.lock:
            state = ServingState(
                model=self.state.model,
                running_requests=self.state.running_requests,
                waiting_requests=self.state.waiting_requests,
                prompt_tokens_total=self.state.prompt_tokens_total,
                generation_tokens_total=self.state.generation_tokens_total,
                successful_requests_total=self.state.successful_requests_total,
                failed_requests_total=self.state.failed_requests_total,
                gpu_cache_usage_perc=self.state.gpu_cache_usage_perc,
                cpu_cache_usage_perc=self.state.cpu_cache_usage_perc,
                started_at=self.state.started_at,
                updated_at=self.state.updated_at,
                ttft_seconds=Histogram(self.state.ttft_seconds.buckets, list(self.state.ttft_seconds.values)),
                e2e_seconds=Histogram(self.state.e2e_seconds.buckets, list(self.state.e2e_seconds.values)),
            )

        labels = f'model_name="{state.model}"'
        success_labels = f'{labels},finished_reason="stop"'
        error_labels = f'{labels},error_type="mock_error"'
        uptime = max(0.0, time.time() - state.started_at)

        lines = [
            "# HELP vllm:num_requests_running Number of requests currently running.",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{{{labels}}} {state.running_requests}",
            "# HELP vllm:num_requests_waiting Number of requests waiting to be scheduled.",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting{{{labels}}} {state.waiting_requests}",
            "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage percentage.",
            "# TYPE vllm:gpu_cache_usage_perc gauge",
            f"vllm:gpu_cache_usage_perc{{{labels}}} {state.gpu_cache_usage_perc:.6f}",
            "# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage percentage.",
            "# TYPE vllm:cpu_cache_usage_perc gauge",
            f"vllm:cpu_cache_usage_perc{{{labels}}} {state.cpu_cache_usage_perc:.6f}",
            "# HELP vllm:prompt_tokens_total Total prompt tokens processed.",
            "# TYPE vllm:prompt_tokens_total counter",
            f"vllm:prompt_tokens_total{{{labels}}} {state.prompt_tokens_total}",
            "# HELP vllm:generation_tokens_total Total generation tokens produced.",
            "# TYPE vllm:generation_tokens_total counter",
            f"vllm:generation_tokens_total{{{labels}}} {state.generation_tokens_total}",
            "# HELP vllm:request_success_total Total successful requests.",
            "# TYPE vllm:request_success_total counter",
            f"vllm:request_success_total{{{success_labels}}} {state.successful_requests_total}",
            "# HELP vllm:request_failure_total Total failed requests.",
            "# TYPE vllm:request_failure_total counter",
            f"vllm:request_failure_total{{{error_labels}}} {state.failed_requests_total}",
            "# HELP vllm:time_to_first_token_seconds Time to first token in seconds.",
            "# TYPE vllm:time_to_first_token_seconds histogram",
            *state.ttft_seconds.render("vllm:time_to_first_token_seconds", labels),
            "# HELP vllm:e2e_request_latency_seconds End-to-end request latency in seconds.",
            "# TYPE vllm:e2e_request_latency_seconds histogram",
            *state.e2e_seconds.render("vllm:e2e_request_latency_seconds", labels),
            "# HELP vllm_practice_demo_uptime_seconds Demo exporter uptime in seconds.",
            "# TYPE vllm_practice_demo_uptime_seconds gauge",
            f"vllm_practice_demo_uptime_seconds{{{labels}}} {uptime:.3f}",
            "",
        ]
        return "\n".join(lines)

    def simulate_forever(self) -> None:
        self._log_event("server_started", "vLLM observability demo started")
        while not self.stop_event.is_set():
            self.handle_mock_request(source="simulated")
            time.sleep(self.interval_seconds)
        self._log_event("server_stopped", "vLLM observability demo stopped")

    def handle_mock_request(self, *, source: str, prompt: str | None = None, max_tokens: int | None = None) -> dict[str, object]:
        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        prompt_tokens = self._estimate_prompt_tokens(prompt) if prompt else self.rng.randint(24, 900)
        generation_tokens = max_tokens or self.rng.randint(16, 220)
        waiting = self.rng.randint(0, 8)
        ttft = max(0.01, self.rng.lognormvariate(-2.2, 0.55))
        decode_seconds = generation_tokens * self.rng.uniform(0.003, 0.018)
        e2e = ttft + decode_seconds

        with self.lock:
            self.state.waiting_requests = waiting
            self.state.running_requests += 1
            self.state.updated_at = time.time()

        time.sleep(min(e2e, 0.15))

        with self.lock:
            self.state.running_requests = max(0, self.state.running_requests - 1)
            self.state.waiting_requests = max(0, waiting - self.rng.randint(0, 2))
            self.state.prompt_tokens_total += prompt_tokens
            self.state.generation_tokens_total += generation_tokens
            self.state.successful_requests_total += 1
            self.state.gpu_cache_usage_perc = min(1.0, 0.08 + self.state.running_requests * 0.04 + self.rng.random() * 0.25)
            self.state.cpu_cache_usage_perc = min(1.0, self.rng.random() * 0.08)
            self.state.ttft_seconds.observe(ttft)
            self.state.e2e_seconds.observe(e2e)
            self.state.updated_at = time.time()

        self._log_event(
            "request_completed",
            "request completed",
            request_id=request_id,
            source=source,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            ttft_ms=round(ttft * 1000, 3),
            e2e_ms=round(e2e * 1000, 3),
            waiting_requests=waiting,
        )
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.state.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Mock vLLM response for observability testing.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generation_tokens,
                "total_tokens": prompt_tokens + generation_tokens,
            },
        }

    def stop(self) -> None:
        self.stop_event.set()

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        return max(1, len(prompt.split()) + len(prompt) // 6)

    def _log_event(self, event: str, message: str, **fields: int | float | str) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "info",
            "component": "vllm-openai-server",
            "event": event,
            "model": self.state.model,
            "message": message,
            **fields,
        }
        self.logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


class VllmDemoHandler(BaseHTTPRequestHandler):
    server_state: ObservableVllmServer

    def do_GET(self) -> None:  # noqa: N802 - http.server API name.
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return
        if self.path == "/metrics":
            self._send_text(self.server_state.metrics_text(), content_type="text/plain; version=0.0.4")
            return
        if self.path == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": self.server_state.state.model, "object": "model"}]})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server API name.
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        payload = json.loads(body.decode("utf-8")) if body else {}
        messages = payload.get("messages", [])
        prompt = " ".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
        response = self.server_state.handle_mock_request(
            source="http",
            prompt=prompt,
            max_tokens=int(max_tokens) if max_tokens else None,
        )
        self._send_json(response)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, body: dict[str, object]) -> None:
        self._send_text(json.dumps(body, ensure_ascii=False), content_type="application/json")

    def _send_text(self, body: str, *, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--log-path", default="logs/vllm/vllm-demo.log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=float, default=0.75, help="Seconds between simulated requests")
    parser.add_argument("--no-simulated-traffic", action="store_true", help="Only emit metrics for explicit HTTP requests")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    observable_server = ObservableVllmServer(
        model=args.model,
        log_path=Path(args.log_path),
        seed=args.seed,
        interval_seconds=args.interval,
    )

    VllmDemoHandler.server_state = observable_server
    http_server = ThreadingHTTPServer((args.host, args.port), VllmDemoHandler)

    def request_stop(signum: int, _frame: object) -> None:
        print(f"received signal {signum}; stopping vLLM observability demo", file=sys.stderr)
        observable_server.stop()
        threading.Thread(target=http_server.shutdown, name="http-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    traffic_thread = None
    if not args.no_simulated_traffic:
        traffic_thread = threading.Thread(target=observable_server.simulate_forever, name="mock-traffic", daemon=True)
        traffic_thread.start()

    print(f"mock vLLM API listening on http://{args.host}:{args.port}")
    print(f"metrics listening on http://{args.host}:{args.port}/metrics")
    print(f"vLLM logs writing to {args.log_path}")

    try:
        http_server.serve_forever()
    finally:
        observable_server.stop()
        if traffic_thread is not None:
            traffic_thread.join(timeout=5)
        http_server.server_close()


if __name__ == "__main__":
    main()
