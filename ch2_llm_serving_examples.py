"""Chapter 2 LLM serving examples.

This file turns the chapter examples into runnable, notebook-style Python cells.
Open it in VS Code/Jupyter to run cells marked with "# %%", or run selected
sections from the command line.

Suggested install:
    pip install torch transformers accelerate matplotlib

Optional extras:
    pip install bertviz vllm

Examples:
    python examples/ch2_llm_serving_examples.py --section config
    python examples/ch2_llm_serving_examples.py --section manual-no-cache --max-new-tokens 20
    python examples/ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 20
    python examples/ch2_llm_serving_examples.py --section vllm-batch
"""

# %%
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
pipeline = None


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_PROMPT = "Write a short introduction about the US capital city."
LONG_PROMPT = """
The history of human communication has moved from oral storytelling to writing,
printing, telegraphy, radio, television, the internet, smartphones, and now
AI-mediated conversation. How might the next wave of communication tools shape
our relationships, societies, and sense of identity?
""".strip()


# %%
# Shared helpers


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


def show_setup() -> None:
    print(
        """
Chapter 2 LLM serving examples

This file is split into runnable sections. The default setup section does not
load any model, so it works before optional ML dependencies are installed.

Install the Hugging Face examples:
    pip install torch transformers accelerate matplotlib

Optional extras:
    pip install bertviz vllm

Try:
    python3 ch2_llm_serving_examples.py --section tokenizer
    python3 ch2_llm_serving_examples.py --section config
    python3 ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 20
    python3 ch2_llm_serving_examples.py --section vllm-config
""".strip()
    )


def optional_dependency(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def require_hf_dependencies() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer, pipeline
    load_local_env()
    if torch is not None:
        return
    if not optional_dependency("torch") or not optional_dependency("transformers"):
        raise RuntimeError(
            "This section needs PyTorch and Hugging Face Transformers. "
            "Install them with `pip install torch transformers accelerate`."
        )

    import torch as torch_module
    from transformers import (
        AutoModelForCausalLM as auto_model_for_causal_lm,
        AutoTokenizer as auto_tokenizer,
        pipeline as hf_pipeline,
    )

    torch = torch_module
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer
    pipeline = hf_pipeline


def get_device() -> torch.device:
    require_hf_dependencies()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    require_hf_dependencies()
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(model_name: str = DEFAULT_MODEL, output_attentions: bool = False):
    require_hf_dependencies()
    device = get_device()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "output_attentions": output_attentions,
    }
    if device.type in {"cuda", "mps"}:
        kwargs["torch_dtype"] = preferred_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(device).eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    logits = logits[:, -1, :]
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def plot_latencies(
    no_cache_times: list[float] | None = None,
    kv_cache_times: list[float] | None = None,
) -> None:
    if not optional_dependency("matplotlib"):
        print("matplotlib is not installed; skipping latency plot.")
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 4))
    if no_cache_times:
        plt.plot(no_cache_times, label="manual generation without KV cache")
    if kv_cache_times:
        plt.plot(kv_cache_times, label="manual generation with KV cache")
    plt.xlabel("Generated token index")
    plt.ylabel("Seconds per token")
    plt.title("Token generation latency")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.show()


# %%
# 1. Tokenizer: raw text -> tokens -> token IDs


def run_tokenizer_demo(model_name: str = DEFAULT_MODEL, text: str = DEFAULT_PROMPT) -> None:
    tokenizer = load_tokenizer(model_name)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"Text: {text}")
    print(f"Token count: {len(tokens)}")
    for i, (token, token_id) in enumerate(zip(tokens, input_ids)):
        print(f"{i:02d} token={token!r:>16} id={token_id}")


# %%
# 2. Inspect Qwen architecture and model configuration


def print_model_config(model_name: str = DEFAULT_MODEL) -> None:
    model = load_model(model_name)
    config = model.config

    print("\n=== Model Configuration Parameters ===")
    print("\nArchitecture Parameters:")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Intermediate size: {config.intermediate_size}")

    print("\nTokenizer Parameters:")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Maximum position embeddings: {config.max_position_embeddings}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel Size:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nSelected Model-specific Parameters:")
    skip_keys = {"architectures", "model_type", "torch_dtype"}
    for key, value in config.to_dict().items():
        if key not in skip_keys:
            print(f"{key}: {value}")


def print_module_structure(module: torch.nn.Module, prefix: str = "", max_depth: int = 4) -> None:
    if max_depth < 0:
        return

    for name, child in module.named_children():
        if name in {"_orig_mod", "wrapped_model"}:
            continue
        print(f"{prefix}{name}: {type(child).__name__}")

        child_type = type(child).__name__.lower()
        if "attention" in child_type or "attn" in name.lower():
            print(f"{prefix}  attention details:")
            for attr in ["num_heads", "num_key_value_heads", "head_dim", "hidden_size"]:
                if hasattr(child, attr):
                    print(f"{prefix}    {attr}: {getattr(child, attr)}")
            if hasattr(child, "rotary_emb"):
                print(f"{prefix}    has_rotary_embeddings: {child.rotary_emb is not None}")

        print_module_structure(child, prefix + "  ", max_depth - 1)


def inspect_decoder_layer(model_name: str = DEFAULT_MODEL, layer_index: int = 0) -> None:
    model = load_model(model_name)
    print("\nModel top-level structure:")
    print_module_structure(model, max_depth=1)

    decoder_layer = model.model.layers[layer_index]
    print(f"\nDecoder layer {layer_index} structure:")
    print_module_structure(decoder_layer, max_depth=3)


# %%
# 3. Attention visualization and fallback summary


def run_attention_demo(model_name: str = DEFAULT_MODEL, text: str = DEFAULT_PROMPT) -> None:
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, output_attentions=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    if optional_dependency("bertviz"):
        from bertviz import head_view

        head_view(attentions, tokens)
        return

    print("bertviz is not installed; showing a textual attention summary instead.")
    last_layer = attentions[-1][0]
    token_index = len(tokens) - 1
    print(f"Last token: {tokens[token_index]!r}")
    for head_index in range(min(4, last_layer.shape[0])):
        weights = last_layer[head_index, token_index]
        top = torch.topk(weights, k=min(5, weights.numel()))
        pairs = [(tokens[i], float(w)) for w, i in zip(top.values, top.indices)]
        print(f"Head {head_index}: {pairs}")


# %%
# 4. Hugging Face pipeline: simple generation API


def run_hf_pipeline(model_name: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT) -> None:
    device = get_device()
    generator = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=preferred_dtype(device),
        device=0 if device.type == "cuda" else -1,
        trust_remote_code=True,
    )
    generated_text = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        num_return_sequences=1,
    )
    print(generated_text[0]["generated_text"])


# %%
# 5. Manual autoregressive generation without KV cache


@dataclass
class GenerationResult:
    text: str
    token_times: list[float]
    token_ids: list[int]


def manual_generate_no_cache(
    model_name: str = DEFAULT_MODEL,
    prompt: str = LONG_PROMPT,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
) -> GenerationResult:
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    idx = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    times: list[float] = []
    new_token_ids: list[int] = []
    start_total = time.time()

    for step in range(max_new_tokens):
        step_start = time.time()
        with torch.no_grad():
            outputs = model(idx)
            idx_next = sample_next_token(outputs.logits, temperature=temperature)
        synchronize_if_needed(model.device)

        token_time = time.time() - step_start
        times.append(token_time)
        token_id = int(idx_next.item())
        new_token_ids.append(token_id)
        print(f"{step:03d} next token: {tokenizer.decode([token_id])!r} ({token_time:.4f}s)")

        idx = torch.cat((idx, idx_next), dim=1)
        if token_id == tokenizer.eos_token_id:
            print("\n[Generation completed: EOS token reached]")
            break

    text = tokenizer.decode(idx[0], skip_special_tokens=True)
    print(f"\nTotal time: {time.time() - start_total:.4f}s")
    print(f"Average time/token: {sum(times) / max(len(times), 1):.4f}s")
    return GenerationResult(text=text, token_times=times, token_ids=new_token_ids)


# %%
# 6. Manual autoregressive generation with KV cache


def manual_generate_with_kv_cache(
    model_name: str = DEFAULT_MODEL,
    prompt: str = LONG_PROMPT,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
) -> GenerationResult:
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    full_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_ids = full_ids
    past_key_values = None
    times: list[float] = []
    new_token_ids: list[int] = []
    start_total = time.time()

    for step in range(max_new_tokens):
        phase = "prefill" if past_key_values is None else "decode"
        step_start = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            idx_next = sample_next_token(outputs.logits, temperature=temperature)
        synchronize_if_needed(model.device)

        token_time = time.time() - step_start
        times.append(token_time)
        token_id = int(idx_next.item())
        new_token_ids.append(token_id)
        print(
            f"{step:03d} {phase:7s} input_shape={tuple(input_ids.shape)} "
            f"next={tokenizer.decode([token_id])!r} ({token_time:.4f}s)"
        )

        input_ids = idx_next
        full_ids = torch.cat((full_ids, idx_next), dim=1)
        if token_id == tokenizer.eos_token_id:
            print("\n[Generation completed: EOS token reached]")
            break

    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    print(f"\nTotal time: {time.time() - start_total:.4f}s")
    print(f"Average time/token: {sum(times) / max(len(times), 1):.4f}s")
    if times:
        print(f"Prefill latency: {times[0]:.4f}s")
        if len(times) > 1:
            print(f"Average decode latency: {sum(times[1:]) / len(times[1:]):.4f}s")
    return GenerationResult(text=text, token_times=times, token_ids=new_token_ids)


def compare_manual_generation(
    model_name: str = DEFAULT_MODEL,
    prompt: str = LONG_PROMPT,
    max_new_tokens: int = 30,
) -> None:
    print("\n=== Without KV cache ===")
    no_cache = manual_generate_no_cache(model_name, prompt, max_new_tokens)
    print("\n=== With KV cache ===")
    kv_cache = manual_generate_with_kv_cache(model_name, prompt, max_new_tokens)
    plot_latencies(no_cache.token_times, kv_cache.token_times)


# %%
# 7. vLLM basic serving example


def require_vllm() -> None:
    if not optional_dependency("vllm"):
        raise RuntimeError(
            "vLLM is not installed in this environment. Install it with `pip install vllm` "
            "on a Linux/CUDA environment, then rerun this section."
        )


def run_vllm_basic(model_name: str = DEFAULT_MODEL) -> None:
    require_vllm()
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, dtype="float16")
    prompt = """
You are an expert AI historian writing a detailed chapter for a book titled
"The Evolution of Human-AI Collaboration." Write in a formal tone, with rich
detail and examples in each era.
""".strip()
    inference_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

    start = time.time()
    outputs = llm.generate([prompt], inference_params)
    print(f"vLLM elapsed time: {time.time() - start:.4f}s")
    for output in outputs:
        print(output.outputs[0].text)


def show_vllm_advanced_config() -> None:
    print(
        """
from vllm import LLM, SamplingParams

model = LLM(
    model="Qwen/Qwen2.5-7B",
    swap_space=16,
    max_model_len=4096,
    block_size=16,
    enable_prefix_caching=True,
    max_num_seqs=256,
    enable_chunked_prefill=True,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_tokens=100,
    stop=["\\n", "###"],
    frequency_penalty=0.1,
    presence_penalty=0.1,
    repetition_penalty=1.1,
    skip_special_tokens=True,
)
""".strip()
    )


# %%
# 8. Hugging Face vs vLLM quick comparison


def compare_hf_and_vllm(model_name: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT) -> None:
    require_vllm()
    from vllm import LLM, SamplingParams

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    start = time.time()
    hf_outputs = generator(prompt, max_new_tokens=128, temperature=0.8, top_p=0.95)
    hf_time = time.time() - start

    llm = LLM(model=model_name, dtype="float16")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

    start = time.time()
    vllm_outputs = llm.generate([prompt], sampling_params)
    vllm_time = time.time() - start

    print(f"Hugging Face elapsed time: {hf_time:.4f}s")
    print(f"vLLM elapsed time: {vllm_time:.4f}s")
    if vllm_time > 0:
        print(f"Speedup: {hf_time / vllm_time:.2f}x")
    print("\nHF output:")
    print(hf_outputs[0]["generated_text"])
    print("\nvLLM output:")
    print(vllm_outputs[0].outputs[0].text)


# %%
# 9. vLLM streaming example


async def run_vllm_streaming(model_name: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT) -> None:
    require_vllm()
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    engine_args = AsyncEngineArgs(model=model_name, dtype="float16")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100, stop=["\n"])
    request_id = f"request-{int(time.time() * 1000)}"
    results_generator = engine.generate(prompt, sampling_params, request_id=request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        for chunk in request_output.outputs:
            print(chunk.text, end="", flush=True)
    print()

    if final_output is not None:
        print(f"\nFinished request: {request_id}")


# %%
# 10. vLLM batch serving example


def run_vllm_batch(model_name: str = DEFAULT_MODEL) -> None:
    require_vllm()
    from vllm import LLM, SamplingParams

    prompts = [
        "What is the meaning of life?",
        "Write a short story about a robot learning to love.",
        "Explain quantum physics in simple terms.",
        "Translate 'Hello, world!' into Spanish.",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    llm = LLM(model=model_name, dtype="float16")

    start = time.time()
    batch_outputs = llm.generate(prompts, sampling_params)
    batch_time = time.time() - start
    print(f"\nvLLM generation time for 4 prompts in a batch: {batch_time:.4f}s")

    start = time.time()
    sequential_outputs = []
    for prompt in prompts:
        sequential_outputs.extend(llm.generate([prompt], sampling_params))
    sequential_time = time.time() - start
    print(f"vLLM generation time for 4 prompts one by one: {sequential_time:.4f}s")
    if batch_time > 0:
        print(f"Batch throughput improvement: {sequential_time / batch_time:.2f}x")

    print("\nBatch outputs:")
    for prompt, output in zip(prompts, batch_outputs):
        print(f"\nPrompt: {prompt}")
        print(output.outputs[0].text)


# %%
# Command-line entrypoint


SECTIONS = {
    "setup": show_setup,
    "tokenizer": run_tokenizer_demo,
    "config": print_model_config,
    "decoder": inspect_decoder_layer,
    "attention": run_attention_demo,
    "pipeline": run_hf_pipeline,
    "manual-no-cache": manual_generate_no_cache,
    "manual-kv-cache": manual_generate_with_kv_cache,
    "compare-manual": compare_manual_generation,
    "vllm-basic": run_vllm_basic,
    "vllm-config": show_vllm_advanced_config,
    "hf-vs-vllm": compare_hf_and_vllm,
    "vllm-batch": run_vllm_batch,
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID")
    parser.add_argument(
        "--section",
        default="setup",
        choices=sorted([*SECTIONS.keys(), "vllm-stream"]),
        help="Which chapter example to run",
    )
    parser.add_argument("--prompt", default=None, help="Prompt override")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    prompt = args.prompt

    if args.section == "vllm-stream":
        asyncio.run(run_vllm_streaming(args.model, prompt or DEFAULT_PROMPT))
        return

    if args.section == "setup":
        show_setup()
    elif args.section in {"manual-no-cache", "manual-kv-cache"}:
        SECTIONS[args.section](
            model_name=args.model,
            prompt=prompt or LONG_PROMPT,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.section == "compare-manual":
        SECTIONS[args.section](
            model_name=args.model,
            prompt=prompt or LONG_PROMPT,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.section in {"tokenizer", "attention", "pipeline", "hf-vs-vllm"}:
        SECTIONS[args.section](model_name=args.model, text=prompt or DEFAULT_PROMPT) if args.section in {
            "tokenizer",
            "attention",
        } else SECTIONS[args.section](model_name=args.model, prompt=prompt or DEFAULT_PROMPT)
    elif args.section == "decoder":
        SECTIONS[args.section](model_name=args.model)
    elif args.section in {"config", "vllm-basic", "vllm-batch"}:
        SECTIONS[args.section](model_name=args.model)
    elif args.section == "vllm-config":
        show_vllm_advanced_config()
    else:
        raise ValueError(f"Unknown section: {args.section}")


if __name__ == "__main__":
    main()
