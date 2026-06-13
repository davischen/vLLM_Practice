"""Chapter 3 model serving system design examples.

This file implements the chapter's serving-system examples as one runnable,
notebook-style Python program. It defaults to a deterministic mock backend so
the architecture can be exercised without downloading a model.

Optional installs:
    pip install fastapi uvicorn httpx
    pip install torch transformers accelerate
    pip install vllm
    pip install tritonclient[http] requests numpy

Examples:
    python3 examples/ch3_model_serving_system_design.py --section basic
    python3 examples/ch3_model_serving_system_design.py --section batch
    python3 examples/ch3_model_serving_system_design.py --section stream
    python3 examples/ch3_model_serving_system_design.py --section multimodel
    python3 examples/ch3_model_serving_system_design.py --section vllm

Serve the FastAPI demo:
    python3 examples/ch3_model_serving_system_design.py --serve --port 8000
"""

# %%
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional


DEFAULT_LLM_MODEL = "facebook/opt-125m"


# %%
# Dependency helpers


def load_local_env(filename: str = ".env") -> None:
    """Load simple KEY=value pairs from .env without overriding real env vars."""
    candidates = [Path.cwd() / filename]
    if "__file__" in globals():
        here = Path(__file__).resolve().parent
        candidates.extend([here / filename, here.parent / filename])

    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def optional_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_dependency(module_name: str, install_hint: str) -> None:
    if not optional_dependency(module_name):
        raise RuntimeError(f"{module_name!r} is not installed. Install it with `{install_hint}`.")


# %%
# Request tracking and workload scheduling


@dataclass
class Sequence:
    """The serving system tracks each prompt as its own schedulable unit."""

    id: str
    prompt: str
    client_stream: Any = None
    loop: Any = None
    output: list[str] = field(default_factory=list)
    finished: bool = False

    @property
    def token_count(self) -> int:
        return len(self.output)

    @property
    def current_text(self) -> str:
        return self.prompt + "".join(self.output)


class WorkloadManager:
    """FIFO scheduler with a fixed maximum batch size."""

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.incoming_queue: queue.Queue[Sequence] = queue.Queue()
        self.active_sequences: list[Sequence] = []
        self.sequence_map: dict[str, Sequence] = {}
        self.lock = threading.Lock()

    def add_request(self, prompt: str) -> str:
        request_id = str(uuid.uuid4())
        sequence = Sequence(request_id, prompt)
        with self.lock:
            self.incoming_queue.put(sequence)
            self.sequence_map[request_id] = sequence
        return request_id

    def add_streaming_request(self, prompt: str, client_stream: Any, loop: Any) -> str:
        request_id = str(uuid.uuid4())
        sequence = Sequence(request_id, prompt, client_stream=client_stream, loop=loop)
        with self.lock:
            self.incoming_queue.put(sequence)
            self.sequence_map[request_id] = sequence
        return request_id

    def get_next_batch(self) -> list[Sequence]:
        with self.lock:
            self.active_sequences = [seq for seq in self.active_sequences if not seq.finished]
            while len(self.active_sequences) < self.batch_size and not self.incoming_queue.empty():
                self.active_sequences.append(self.incoming_queue.get())
            return list(self.active_sequences)

    def update_batch_results(self, results: list[dict[str, Any]]) -> None:
        with self.lock:
            for result in results:
                sequence = self.sequence_map[result["request_id"]]
                sequence.output = [result["generated_text"]]
                sequence.finished = True
            self.active_sequences = [seq for seq in self.active_sequences if not seq.finished]

    def update_sequence_output(self, request_id: str, token: str) -> None:
        with self.lock:
            self.sequence_map[request_id].output.append(token)

    def mark_finished(self, request_id: str) -> None:
        with self.lock:
            if request_id in self.sequence_map:
                self.sequence_map[request_id].finished = True
            self.active_sequences = [seq for seq in self.active_sequences if seq.id != request_id]

    def get_sequence(self, request_id: str) -> Sequence:
        with self.lock:
            return self.sequence_map[request_id]

    def remove_finished_sequence(self, request_id: str) -> None:
        with self.lock:
            self.sequence_map.pop(request_id, None)
            self.active_sequences = [seq for seq in self.active_sequences if seq.id != request_id]

    def is_finished(self, request_ids: list[str]) -> bool:
        with self.lock:
            return all(self.sequence_map[request_id].finished for request_id in request_ids)


# %%
# Model backends


class MockTextBackend:
    """A tiny deterministic backend for demonstrating scheduling mechanics."""

    token_plan = [" a", " compact", " model", " serving", " demo", "."]

    def generate_batch(self, sequences: list[Sequence], max_tokens: int) -> list[dict[str, Any]]:
        time.sleep(0.05)
        results = []
        for seq in sequences:
            suffix = "".join(self.token_plan[:max_tokens])
            results.append(
                {
                    "request_id": seq.id,
                    "generated_text": f"{seq.prompt}{suffix}",
                }
            )
        return results

    def forward_batch(self, sequences: list[Sequence], max_tokens: int) -> list[dict[str, Any]]:
        time.sleep(0.05)
        results = []
        for seq in sequences:
            token_index = seq.token_count
            token = self.token_plan[token_index % len(self.token_plan)]
            is_finished = token_index + 1 >= min(max_tokens, len(self.token_plan))
            results.append(
                {
                    "request_id": seq.id,
                    "token": token,
                    "is_finished": is_finished,
                }
            )
        return results


class TransformersTextBackend:
    """Optional real Hugging Face backend for small causal language models."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        load_local_env()
        require_dependency("torch", "pip install torch transformers accelerate")
        require_dependency("transformers", "pip install torch transformers accelerate")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_batch(self, sequences: list[Sequence], max_tokens: int) -> list[dict[str, Any]]:
        prompts = [seq.prompt for seq in sequences]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [
            {"request_id": seq.id, "generated_text": text}
            for seq, text in zip(sequences, texts)
        ]

    def forward_batch(self, sequences: list[Sequence], max_tokens: int) -> list[dict[str, Any]]:
        results = []
        generated = self.generate_batch(sequences, max_tokens=1)
        for seq, result in zip(sequences, generated):
            token = result["generated_text"][len(seq.prompt) :] or " "
            results.append(
                {
                    "request_id": seq.id,
                    "token": token,
                    "is_finished": seq.token_count + 1 >= max_tokens,
                }
            )
        return results


def make_text_backend(name: str, model_name: str = DEFAULT_LLM_MODEL):
    if name == "mock":
        return MockTextBackend()
    if name == "transformers":
        return TransformersTextBackend(model_name)
    raise ValueError(f"Unknown backend: {name}")


# %%
# Model worker and executor


class ModelWorker:
    """Runs model inference. In production this often lives in a GPU-bound process."""

    def __init__(self, backend_name: str = "mock", model_name: str = DEFAULT_LLM_MODEL):
        self.backend = make_text_backend(backend_name, model_name)

    def generate(self, sequences: list[Sequence], max_tokens: int) -> list[dict[str, Any]]:
        return self.backend.generate_batch(sequences, max_tokens)

    def forward(self, sequences: list[Sequence], max_tokens: int) -> list[dict[str, Any]]:
        return self.backend.forward_batch(sequences, max_tokens)

    @staticmethod
    def run(
        backend_name: str,
        model_name: str,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        max_tokens: int,
    ) -> None:
        worker = ModelWorker(backend_name, model_name)
        while True:
            request = task_queue.get()
            if request["type"] == "shutdown":
                break
            if request["type"] == "generate":
                result_queue.put(("complete", worker.generate(request["sequences"], max_tokens)))
            elif request["type"] == "forward":
                result_queue.put(("tokens", worker.forward(request["sequences"], max_tokens)))


class ModelExecutor:
    """Coordinates calls into the model worker."""

    def __init__(
        self,
        backend_name: str = "mock",
        model_name: str = DEFAULT_LLM_MODEL,
        max_tokens: int = 20,
        use_process: bool = False,
    ):
        self.backend_name = backend_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.use_process = use_process
        self.worker: Optional[ModelWorker] = None
        self.task_queue: Optional[mp.Queue] = None
        self.result_queue: Optional[mp.Queue] = None
        self.worker_process: Optional[mp.Process] = None
        self.setup_worker()

    def setup_worker(self) -> None:
        if self.use_process:
            self.task_queue = mp.Queue()
            self.result_queue = mp.Queue()
            self.worker_process = mp.Process(
                target=ModelWorker.run,
                args=(
                    self.backend_name,
                    self.model_name,
                    self.task_queue,
                    self.result_queue,
                    self.max_tokens,
                ),
            )
            self.worker_process.start()
        else:
            self.worker = ModelWorker(self.backend_name, self.model_name)

    def execute_batch(self, sequences: list[Sequence]) -> list[dict[str, Any]]:
        if not sequences:
            return []
        if self.use_process:
            self.task_queue.put({"type": "generate", "sequences": sequences})
            _, results = self.result_queue.get()
            return results
        return self.worker.generate(sequences, self.max_tokens)

    def execute_forward_batch(self, sequences: list[Sequence]) -> list[dict[str, Any]]:
        if not sequences:
            return []
        if self.use_process:
            self.task_queue.put({"type": "forward", "sequences": sequences})
            _, results = self.result_queue.get()
            return results
        return self.worker.forward(sequences, self.max_tokens)

    def shutdown(self) -> None:
        if self.use_process and self.worker_process is not None:
            self.task_queue.put({"type": "shutdown"})
            self.worker_process.join(timeout=3)


# %%
# LLM engine: single request, batch request, and streaming request orchestration


class LLMEngine:
    def __init__(
        self,
        backend_name: str = "mock",
        model_name: str = DEFAULT_LLM_MODEL,
        batch_size: int = 4,
        max_tokens: int = 6,
        use_process: bool = False,
    ):
        self.max_tokens = max_tokens
        self.workload_manager = WorkloadManager(batch_size=batch_size)
        self.model_executor = ModelExecutor(
            backend_name=backend_name,
            model_name=model_name,
            max_tokens=max_tokens,
            use_process=use_process,
        )
        self._stop = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None

    def basic_generate(self, prompt: str) -> str:
        return self.generate([prompt])[0]

    def generate(self, prompts: list[str]) -> list[str]:
        request_ids = [self.workload_manager.add_request(prompt) for prompt in prompts]

        while not self.workload_manager.is_finished(request_ids):
            sequences = self.workload_manager.get_next_batch()
            if not sequences:
                time.sleep(0.005)
                continue
            results = self.model_executor.execute_batch(sequences)
            self.workload_manager.update_batch_results(results)

        generated_texts = []
        for request_id in request_ids:
            sequence = self.workload_manager.get_sequence(request_id)
            generated_texts.append(sequence.output[0])
            self.workload_manager.remove_finished_sequence(request_id)
        return generated_texts

    def start_processing_loop(self) -> None:
        if self._processing_thread and self._processing_thread.is_alive():
            return
        self._processing_thread = threading.Thread(
            target=self.requests_processing_loop,
            name="llm-streaming-batch-loop",
            daemon=True,
        )
        self._processing_thread.start()

    def requests_processing_loop(self) -> None:
        while not self._stop.is_set():
            active_sequences = self.workload_manager.get_next_batch()
            streaming_sequences = [seq for seq in active_sequences if seq.client_stream is not None]
            if not streaming_sequences:
                time.sleep(0.01)
                continue

            tokens = self.model_executor.execute_forward_batch(streaming_sequences)
            for token in tokens:
                sequence_id = token["request_id"]
                sequence = self.workload_manager.get_sequence(sequence_id)
                data = json.dumps({"token": token["token"], "sequence_id": sequence_id})
                asyncio.run_coroutine_threadsafe(sequence.client_stream.put(data), sequence.loop)
                self.workload_manager.update_sequence_output(sequence_id, token["token"])

                if token["is_finished"] or sequence.token_count >= self.max_tokens:
                    asyncio.run_coroutine_threadsafe(sequence.client_stream.put(None), sequence.loop)
                    self.workload_manager.mark_finished(sequence_id)

    async def event_generator(self, prompt: str) -> AsyncIterator[str]:
        self.start_processing_loop()
        loop = asyncio.get_running_loop()
        client_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        sequence_id = self.workload_manager.add_streaming_request(prompt, client_queue, loop)
        try:
            while True:
                data = await client_queue.get()
                if data is None:
                    break
                yield f"data: {data}\n\n"
        finally:
            self.workload_manager.remove_finished_sequence(sequence_id)

    def shutdown(self) -> None:
        self._stop.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2)
        self.model_executor.shutdown()


# %%
# Optional FastAPI service matching the chapter's API examples


def create_single_model_app(engine: Optional[LLMEngine] = None):
    require_dependency("fastapi", "pip install fastapi uvicorn")

    from fastapi import Body, Depends, FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="Chapter 3 single-model serving demo")
    llm_engine = engine or LLMEngine()

    def get_llm() -> LLMEngine:
        return llm_engine

    @app.get("/")
    async def root():
        return {
            "name": "Chapter 3 single-model serving demo",
            "status": "ok",
            "endpoints": {
                "docs": "GET /docs",
                "basic_generate": "POST /basic_generate",
                "generate": "POST /generate",
                "generate_stream": "POST /generate_stream",
            },
            "examples": {
                "basic_generate_body": {"prompt": "Hello, I am"},
                "generate_body": {"prompts": ["Hello, I am", "The weather is"]},
            },
        }

    @app.post("/basic_generate")
    async def basic_generate(request: dict[str, Any] = Body(...), llm: LLMEngine = Depends(get_llm)):
        generated_text = llm.basic_generate(str(request["prompt"]))
        return {"generated_text": generated_text}

    @app.post("/generate")
    async def generate(request: dict[str, Any] = Body(...), llm: LLMEngine = Depends(get_llm)):
        prompts = [str(prompt) for prompt in request["prompts"]]
        generated_texts = llm.generate(prompts)
        return {"generated_texts": generated_texts}

    @app.post("/generate_stream")
    async def generate_stream(request: dict[str, Any] = Body(...), llm: LLMEngine = Depends(get_llm)):
        return StreamingResponse(
            llm.event_generator(str(request["prompt"])),
            media_type="text/event-stream",
        )

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        llm_engine.shutdown()

    return app


# %%
# vLLM-backed serving engine, as a compact contrast to the manual system


class VLLMEngine:
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, max_tokens: int = 20):
        require_dependency("vllm", "pip install vllm")
        from vllm import LLM, SamplingParams

        self.max_tokens = max_tokens
        self.sampling_params_cls = SamplingParams
        self.vllm_model = LLM(model=model_name)

    def generate_vllm(self, prompts: list[str]) -> list[str]:
        sampling_params = self.sampling_params_cls(
            temperature=0.7,
            top_p=0.95,
            max_tokens=self.max_tokens,
        )
        outputs = self.vllm_model.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


# %%
# Multi-model serving: metadata store, workers, LRU manager


@dataclass
class ModelMetadata:
    id: str
    name: str
    type: str
    framework: str
    version: str
    description: str


class ModelStore:
    def __init__(self, config_path: Optional[str] = None):
        self.models: dict[str, ModelMetadata] = {}
        if config_path:
            self._load_config(config_path)
        else:
            self._load_default_config()

    def _load_config(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for model in config["models"]:
            self.models[model["id"]] = ModelMetadata(**model)

    def _load_default_config(self) -> None:
        defaults = [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
                "type": "text",
                "framework": "mock_transformers",
                "version": "1.0.0",
                "description": "Sentiment analysis model",
            },
            {
                "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "name": "pytorch/vision:mobilenet_v2",
                "type": "image",
                "framework": "mock_torchvision",
                "version": "1.0.0",
                "description": "Image classification model",
            },
            {
                "id": "11111111-2222-3333-4444-555555555555",
                "name": "demo-regression-model",
                "type": "tabular",
                "framework": "mock_regression",
                "version": "1.0.0",
                "description": "Numeric regression model",
            },
        ]
        for model in defaults:
            self.models[model["id"]] = ModelMetadata(**model)

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self.models.get(model_id)


class GenericModelWorker:
    def __init__(self, model_metadata: ModelMetadata):
        self.model_metadata = model_metadata
        self.loaded_at = time.time()
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        self.model = {"name": self.model_metadata.name}

    def predict(self, input_data: Any) -> dict[str, Any]:
        raise NotImplementedError


class MockTransformerWorker(GenericModelWorker):
    def predict(self, input_data: Any) -> dict[str, Any]:
        text = str(input_data)
        positive_words = {"great", "love", "excellent", "enjoyed", "good"}
        score = sum(word in text.lower() for word in positive_words)
        positive = min(0.95, 0.35 + 0.2 * score)
        return {
            "model_id": self.model_metadata.id,
            "label": "POSITIVE" if positive >= 0.5 else "NEGATIVE",
            "predictions": [[round(1 - positive, 4), round(positive, 4)]],
        }


class MockTorchVisionWorker(GenericModelWorker):
    def predict(self, input_data: Any) -> dict[str, Any]:
        return {
            "model_id": self.model_metadata.id,
            "top_class": "demo_object",
            "predictions": [{"label": "demo_object", "score": 0.88}],
            "input_summary": str(input_data)[:80],
        }


class MockRegressionWorker(GenericModelWorker):
    def predict(self, input_data: Any) -> dict[str, Any]:
        values = input_data if isinstance(input_data, list) else [input_data]
        numeric = [float(x) for x in values]
        return {
            "model_id": self.model_metadata.id,
            "prediction": round(sum(numeric) / max(len(numeric), 1), 4),
        }


class ModelEngine:
    def __init__(self):
        self.workers: dict[str, GenericModelWorker] = {}

    def create_worker(self, model_metadata: ModelMetadata) -> GenericModelWorker:
        if model_metadata.id in self.workers:
            return self.workers[model_metadata.id]

        if model_metadata.framework == "mock_transformers":
            worker = MockTransformerWorker(model_metadata)
        elif model_metadata.framework == "mock_torchvision":
            worker = MockTorchVisionWorker(model_metadata)
        elif model_metadata.framework == "mock_regression":
            worker = MockRegressionWorker(model_metadata)
        elif model_metadata.framework == "triton":
            worker = TritonWorker(model_metadata)
        else:
            raise ValueError(f"Unsupported framework: {model_metadata.framework}")

        self.workers[model_metadata.id] = worker
        return worker

    def get_worker(self, model_id: str) -> Optional[GenericModelWorker]:
        return self.workers.get(model_id)

    def delete_worker(self, model_id: str) -> None:
        self.workers.pop(model_id, None)


class ModelManager:
    def __init__(self, model_store: ModelStore, max_models: int = 2):
        self.model_store = model_store
        self.max_models = max_models
        self.model_cache: OrderedDict[str, GenericModelWorker] = OrderedDict()
        self.model_engine = ModelEngine()

    def get_model_worker(self, model_id: str) -> Optional[GenericModelWorker]:
        if model_id in self.model_cache:
            self.model_cache.move_to_end(model_id)
            return self.model_engine.get_worker(model_id)

        model_metadata = self.model_store.get_model(model_id)
        if model_metadata is None:
            return None

        if len(self.model_cache) >= self.max_models:
            evicted_id, _ = self.model_cache.popitem(last=False)
            self.model_engine.delete_worker(evicted_id)
            print(f"Evicted least recently used model: {evicted_id}")

        worker = self.model_engine.create_worker(model_metadata)
        self.model_cache[model_id] = worker
        return worker

    def cache_state(self) -> list[str]:
        return list(self.model_cache.keys())


def create_multi_model_app(model_manager: Optional[ModelManager] = None):
    require_dependency("fastapi", "pip install fastapi uvicorn")

    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="Chapter 3 multi-model serving demo")
    manager = model_manager or ModelManager(ModelStore(), max_models=2)

    @app.get("/")
    async def root():
        return {
            "name": "Chapter 3 multi-model serving demo",
            "status": "ok",
            "endpoints": {
                "docs": "GET /docs",
                "predict": "POST /predict",
            },
            "example_model_ids": {
                "text_classification": "550e8400-e29b-41d4-a716-446655440000",
                "image_classification": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "regression": "11111111-2222-3333-4444-555555555555",
            },
        }

    @app.post("/predict")
    async def predict(request: dict[str, Any] = Body(...)):
        model_id = str(request["model_id"])
        worker = manager.get_model_worker(model_id)
        if worker is None:
            raise HTTPException(status_code=404, detail=f"Unknown model_id: {model_id}")
        return worker.predict(request["input_data"])

    return app


# %%
# Triton wrapper. This mirrors the chapter code but requires a running Triton server.


class TritonWorker(GenericModelWorker):
    def __init__(self, model_metadata: ModelMetadata, triton_url: str = "0.0.0.0:8009"):
        self.triton_url = triton_url
        require_dependency("requests", "pip install requests tritonclient[http] numpy")
        require_dependency("numpy", "pip install requests tritonclient[http] numpy")
        require_dependency("tritonclient.http", "pip install requests tritonclient[http] numpy")

        import requests
        import tritonclient.http as httpclient

        self.requests = requests
        self.httpclient = httpclient
        self.client = httpclient.InferenceServerClient(url=self.triton_url)
        super().__init__(model_metadata)

    def _load_model(self) -> None:
        load_url = f"http://{self.triton_url}/v2/repository/models/{self.model_metadata.name}/load"
        response = self.requests.post(load_url, timeout=30)
        response.raise_for_status()
        self.model = {"loaded_in_triton": True}

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        inputs = []
        for name, data in input_data.items():
            if isinstance(data, dict):
                array = np.array(data["data"], dtype=np.float32).reshape(data["shape"])
            else:
                array = np.array(data, dtype=np.float32)
            input_tensor = self.httpclient.InferInput(name, array.shape, "FP32")
            input_tensor.set_data_from_numpy(array)
            inputs.append(input_tensor)

        output_name = input_data.get("output_name", "fc6_1") if isinstance(input_data, dict) else "fc6_1"
        response = self.client.infer(
            model_name=self.model_metadata.name,
            inputs=inputs,
            outputs=[self.httpclient.InferRequestedOutput(output_name)],
        )
        return {output_name: response.as_numpy(output_name).tolist()}

    def unload(self) -> None:
        unload_url = f"http://{self.triton_url}/v2/repository/models/{self.model_metadata.name}/unload"
        self.requests.post(unload_url, timeout=30)

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


# %%
# CLI demos


def demo_basic(backend: str, model_name: str, max_tokens: int) -> None:
    engine = LLMEngine(backend_name=backend, model_name=model_name, max_tokens=max_tokens)
    try:
        print(engine.basic_generate("Hello, I am"))
    finally:
        engine.shutdown()


def demo_batch(backend: str, model_name: str, max_tokens: int) -> None:
    engine = LLMEngine(backend_name=backend, model_name=model_name, max_tokens=max_tokens, batch_size=4)
    prompts = [
        "Hello, I am",
        "The weather is",
        "I want to",
        "The best way to",
        "The most efficient way to",
    ]
    try:
        outputs = engine.generate(prompts)
        for prompt, output in zip(prompts, outputs):
            print(f"\nPrompt: {prompt}")
            print(f"Output: {output}")
    finally:
        engine.shutdown()


async def demo_stream(backend: str, model_name: str, max_tokens: int) -> None:
    engine = LLMEngine(backend_name=backend, model_name=model_name, max_tokens=max_tokens, batch_size=4)

    async def consume(prompt: str) -> None:
        print(f"\nstream start: {prompt}")
        async for event in engine.event_generator(prompt):
            print(event.strip())
        print(f"stream end: {prompt}")

    try:
        await asyncio.gather(
            consume("Hello, I am"),
            consume("I want to"),
            consume("The best way to"),
        )
    finally:
        engine.shutdown()


def demo_multimodel() -> None:
    manager = ModelManager(ModelStore(), max_models=2)
    requests = [
        ("550e8400-e29b-41d4-a716-446655440000", "This movie was great! I really enjoyed it."),
        ("7c9e6679-7425-40de-944b-e07fc1f90ae7", {"image": "pretend image bytes"}),
        ("550e8400-e29b-41d4-a716-446655440000", "This was not good."),
        ("11111111-2222-3333-4444-555555555555", [1, 2, 3, 4]),
    ]

    for model_id, input_data in requests:
        worker = manager.get_model_worker(model_id)
        print(f"\nCache after loading {model_id}: {manager.cache_state()}")
        print(worker.predict(input_data))


def demo_vllm(model_name: str, max_tokens: int) -> None:
    engine = VLLMEngine(model_name=model_name, max_tokens=max_tokens)
    prompts = ["Hello, I am", "The weather is", "I want to"]
    outputs = engine.generate_vllm(prompts)
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(output)


def show_triton_commands() -> None:
    print(
        """
# Load a model into Triton:
curl -X POST http://localhost:8000/v2/repository/models/densenet_onnx/load

# Run inference through Triton:
curl -X POST http://localhost:8000/v2/models/densenet_onnx/infer

# In this file, TritonWorker wraps those management/inference APIs so a
# multi-model service can keep its own model metadata, cache policy, and public
# /predict interface while delegating model execution to Triton.
""".strip()
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=["basic", "batch", "stream", "multimodel", "vllm", "triton"],
        default="basic",
    )
    parser.add_argument("--backend", choices=["mock", "transformers"], default="mock")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI single-model demo")
    parser.add_argument("--multi-serve", action="store_true", help="Run the FastAPI multi-model demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def serve_app(app: Any, host: str, port: int) -> None:
    require_dependency("uvicorn", "pip install fastapi uvicorn")
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if args.serve:
        app = create_single_model_app(
            LLMEngine(backend_name=args.backend, model_name=args.model, max_tokens=args.max_tokens)
        )
        serve_app(app, args.host, args.port)
        return

    if args.multi_serve:
        serve_app(create_multi_model_app(), args.host, args.port)
        return

    if args.section == "basic":
        demo_basic(args.backend, args.model, args.max_tokens)
    elif args.section == "batch":
        demo_batch(args.backend, args.model, args.max_tokens)
    elif args.section == "stream":
        asyncio.run(demo_stream(args.backend, args.model, args.max_tokens))
    elif args.section == "multimodel":
        demo_multimodel()
    elif args.section == "vllm":
        demo_vllm(args.model, args.max_tokens)
    elif args.section == "triton":
        show_triton_commands()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
