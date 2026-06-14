"""vLLM-backed chain-of-thought inference for Homework 3.

This module mirrors the CoT prompting style in ``cot.py`` but uses vLLM for
batched generation. It is intended for local inference/data generation, not for
the course grader path that imports the Hugging Face ``BaseLLM`` classes.

Examples, from the ``homework3`` directory:
    python -m homework.vllm_cot test --max_question 100
    python -m homework.vllm_cot generate_dataset --output_json data/rft_vllm.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import overload


DEFAULT_CHECKPOINT = "HuggingFaceTB/SmolLM2-360M-Instruct"


class VLLMCoTModel:
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vllm is required for homework.vllm_cot. Install it in a Linux/CUDA "
                "environment with a command such as `pip install vllm`."
            ) from exc

        engine_kwargs = {
            "model": checkpoint,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs

        self.llm = LLM(**engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def format_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in unit conversions and quantitative reasoning.\n"
                    "Always think through the calculation step by step before giving the answer.\n"
                    "Double-check your arithmetic and unit consistency.\n"
                    "You must answer EXACTLY in the following format and nothing else:\n"
                    "Conversion: [rate]\n"
                    "Calculation: [math]\n"
                    "<answer>[result]</answer>"
                    "Check that you are not using extra unit conversions beyond what is necessary.\n"
                )
            },
            # 直接教它 kB 到 bit 的最終倍率 (1000 * 8 = 8000)，不解釋原因
            {
                "role": "user",
                "content": "Convert the measurement of 2 m/s into in/s."
            },
            {
                "role": "assistant",
                "content": (
                    "Conversion: 1 m = 39.3700787 in. Since time is the same, 1 m/s = 39.3700787 in/s.\n"
                    "Calculation: 2 * 39.3700787 = 78.7401574\n"
                    "<answer>78.7401574</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 3 kB into bit."
            },
            {
                "role": "assistant",
                "content": (
                    "Conversion: 1 kB = 8000 bit\n"
                    "Calculation: 3 * 8000 = 24000\n"
                    "<answer>24000</answer>"
                )
            },
            # 直接教它 mi/h 到 m/s 的最終常數，不解釋分子分母
            {
                "role": "user",
                "content": "How do we translate 5 mi/h into m/s?"
            },
            {
                "role": "assistant",
                "content": (
                    "Conversion: 1 mi/h = 0.44704 m/s\n"
                    "Calculation: 5 * 0.44704 = 2.2352\n"
                    "<answer>2.2352</answer>"
                )
            },
             {"role": "user", "content": "What is the equivalent of 9 milliliter in mm^3?"},
            {
                "role": "assistant",
                "content": (
                    "Conversion used: 1000 mm^3/milliliter\n"
                    "Calculation: 9 * 1000 = 9000\n"
                    "<answer>9000</answer>"
                ),
            },
            # 直接教它 decade 到 month 的最終倍率 (10 * 12 = 120)
            {
                "role": "user",
                "content": "What is the conversion from decades to month for 2 units?"
            },
            {
                "role": "assistant",
                "content": (
                    "Conversion: 1 decade = 120 month\n"
                    "Calculation: 2 * 120 = 240\n"
                    "<answer>240</answer>"
                )
            },
            # 實際的測試問題
            {
                "role": "user",
                "content": question
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def parse_answer(self, answer: str) -> float:
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str, *, temperature: float = 0, max_tokens: int = 75) -> str:
        return self.batched_generate([prompt], temperature=temperature, max_tokens=max_tokens)[0]

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: None = None,
        temperature: float = 0,
        max_tokens: int = 75,
        top_p: float = 1.0,
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int,
        temperature: float = 0,
        max_tokens: int = 75,
        top_p: float = 1.0,
    ) -> list[list[str]]:
        ...

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
        max_tokens: int = 75,
        top_p: float = 1.0,
    ) -> list[str] | list[list[str]]:
        from vllm import SamplingParams

        n = num_return_sequences or 1
        formatted_prompts = [self.format_prompt(prompt) for prompt in prompts]
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        request_outputs = self.llm.generate(formatted_prompts, sampling_params)

        if num_return_sequences is None:
            return [request_output.outputs[0].text for request_output in request_outputs]
        return [[completion.text for completion in request_output.outputs] for request_output in request_outputs]

    def answer(self, *questions: str) -> list[float]:
        generations = self.batched_generate(list(questions))
        return [self.parse_answer(generation) for generation in generations]


def load() -> VLLMCoTModel:
    return VLLMCoTModel()


def test_model(max_question: int = 100, split: str = "valid", checkpoint: str = DEFAULT_CHECKPOINT) -> None:
    from .data import Dataset, benchmark

    testset = Dataset(split)
    model = VLLMCoTModel(checkpoint=checkpoint)
    benchmark_result = benchmark(model, testset, max_question)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


def generate_dataset(
    output_json: str = "data/rft_vllm.json",
    *,
    split: str = "train",
    max_samples: int | None = None,
    oversample: int = 10,
    temperature: float = 0.8,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> None:
    """Generate RFT-style CoT data with vLLM rejection sampling."""

    from tqdm import tqdm

    from .data import Dataset, is_answer_valid

    model = VLLMCoTModel(checkpoint=checkpoint)
    dataset = Dataset(split)
    count = min(len(dataset), max_samples or len(dataset))
    questions = [dataset[i][0] for i in range(count)]
    correct_answers = [dataset[i][1] for i in range(count)]

    result: list[list[str | float]] = []

    greedy_generations = model.batched_generate(questions, temperature=0)
    misses: list[tuple[int, str, float]] = []
    for idx, generation in enumerate(greedy_generations):
        pred = model.parse_answer(generation)
        if is_answer_valid(pred, correct_answers[idx]):
            result.append([questions[idx], correct_answers[idx], generation])
        else:
            misses.append((idx, questions[idx], correct_answers[idx]))

    if misses and oversample > 0:
        sampled_questions = [question for _, question, _ in misses]
        sampled_generations = model.batched_generate(
            sampled_questions,
            num_return_sequences=oversample,
            temperature=temperature,
        )

        for (idx, question, correct_answer), generations in tqdm(
            zip(misses, sampled_generations),
            total=len(misses),
            desc="Selecting correct vLLM CoT samples",
        ):
            for generation in generations:
                pred = model.parse_answer(generation)
                if is_answer_valid(pred, correct_answer):
                    result.append([question, correct_answer, generation])
                    break

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"saved {len(result)} / {count} accepted CoT samples to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "generate_dataset": generate_dataset, "load": load})
