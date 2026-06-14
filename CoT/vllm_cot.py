"""Standalone vLLM chain-of-thought inference for Homework 3.

This file does not import the ``homework`` package. It can be kept next to the
``data`` directory and run directly even if ``homework3/homework`` is removed.

Examples, from the ``homework3`` directory:
    python vllm_cot.py test --max-question 100
    python vllm_cot.py answer "How many gram are there per 6 kg?"
    python vllm_cot.py generate_dataset --output-json data/rft_vllm.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, overload


DEFAULT_CHECKPOINT = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_DIR = Path(__file__).parent / "data"


class Dataset:
    def __init__(self, split: str, data_dir: str | Path = DATA_DIR):
        self.path = Path(data_dir) / f"{split}.json"
        with self.path.open() as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> list[object]:
        return self.data[idx]


@dataclass
class BenchmarkResult:
    accuracy: float
    answer_rate: float


def is_answer_valid(answer: float, correct_answer: float, relative_tolerance: float = 0.05) -> bool:
    return abs(round(answer, 3) - round(correct_answer, 3)) < relative_tolerance * abs(round(correct_answer, 3))


def benchmark(model: "VLLMCoTModel", dataset: Dataset, max_question: int) -> BenchmarkResult:
    count = min(len(dataset), max_question)
    questions = [str(dataset[i][0]) for i in range(count)]
    correct_answers = [float(dataset[i][1]) for i in range(count)]
    answers = model.answer(*questions)
    valid_answers = [answer for answer in answers if not math.isnan(answer)]
    correct = sum(is_answer_valid(answer, correct) for answer, correct in zip(answers, correct_answers))
    return BenchmarkResult(accuracy=correct / count, answer_rate=len(valid_answers) / count)


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
                "vllm is required for this script. Install it in a Linux/CUDA "
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
                    "You are a helpful assistant that performs unit conversions. "
                    "Be concise. Use step-by-step reasoning. Format your final "
                    "answer inside <answer> tags."
                ),
            },
            {
                "role": "user",
                "content": "How many MB are there per 9 GB?",
            },
            {
                "role": "assistant",
                "content": (
                    "To convert GB to MB, we use the conversion: 1 GB = 1000 MB.\n"
                    "So, 9 GB * 1000 = 9000.0 MB.\n"
                    "<answer>9000.0</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How do we express 4 pound in terms of ounce?",
            },
            {
                "role": "assistant",
                "content": "1 pound = 16 ounces.\nSo, 4 pounds * 16 = 64.0 ounces.\n<answer>64.0</answer>",
            },
            {
                "role": "user",
                "content": "Can you change 9 kmh to its equivalent in mi/h",
            },
            {
                "role": "assistant",
                "content": (
                    "To convert kmh to mi/h, we use the conversion: 1 km = 0.6213711922373.\n"
                    "So, 9 kmh * 0.6213711922373 = 5.592340730136007 mi/h.\n"
                    "<answer>5.592340730136007</answer>"
                ),
            },
            {
                "role": "user",
                "content": "What is the equivalent of 10 century in year?",
            },
            {
                "role": "assistant",
                "content": "1 century = 100 years.\nSo, 10 centuries * 100 = 1000.0 years.\n<answer>1000.0</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        sampling_params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        request_outputs = self.llm.generate(formatted_prompts, sampling_params)

        if num_return_sequences is None:
            return [request_output.outputs[0].text for request_output in request_outputs]
        return [[completion.text for completion in request_output.outputs] for request_output in request_outputs]

    def answer(self, *questions: str) -> list[float]:
        generations = self.batched_generate(list(questions))
        return [self.parse_answer(generation) for generation in generations]


def build_model(args: argparse.Namespace) -> VLLMCoTModel:
    return VLLMCoTModel(
        checkpoint=args.checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )


def cmd_answer(args: argparse.Namespace) -> None:
    model = build_model(args)
    for question, answer in zip(args.questions, model.answer(*args.questions)):
        print(json.dumps({"question": question, "answer": answer}, ensure_ascii=False))


def cmd_test(args: argparse.Namespace) -> None:
    model = build_model(args)
    dataset = Dataset(args.split, args.data_dir)
    result = benchmark(model, dataset, args.max_question)
    print(f"{result.accuracy=}  {result.answer_rate=}")


def cmd_generate_dataset(args: argparse.Namespace) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    model = build_model(args)
    dataset = Dataset(args.split, args.data_dir)
    count = min(len(dataset), args.max_samples or len(dataset))
    questions = [str(dataset[i][0]) for i in range(count)]
    correct_answers = [float(dataset[i][1]) for i in range(count)]
    result: list[list[str | float]] = []

    greedy_generations = model.batched_generate(questions, temperature=0, max_tokens=args.max_tokens)
    misses: list[tuple[int, str, float]] = []
    for idx, generation in enumerate(greedy_generations):
        pred = model.parse_answer(generation)
        if is_answer_valid(pred, correct_answers[idx]):
            result.append([questions[idx], correct_answers[idx], generation])
        else:
            misses.append((idx, questions[idx], correct_answers[idx]))

    if misses and args.oversample > 0:
        sampled_questions = [question for _, question, _ in misses]
        sampled_generations = model.batched_generate(
            sampled_questions,
            num_return_sequences=args.oversample,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        iterator: Iterable[tuple[tuple[int, str, float], Sequence[str]]] = zip(misses, sampled_generations)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(misses), desc="Selecting correct vLLM CoT samples")
        for (_idx, question, correct_answer), generations in iterator:
            for generation in generations:
                pred = model.parse_answer(generation)
                if is_answer_valid(pred, correct_answer):
                    result.append([question, correct_answer, generation])
                    break

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {len(result)} / {count} accepted CoT samples to {args.output_json}")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    answer_parser = subparsers.add_parser("answer", help="Answer one or more unit conversion questions")
    add_model_args(answer_parser)
    answer_parser.add_argument("questions", nargs="+")
    answer_parser.set_defaults(func=cmd_answer)

    test_parser = subparsers.add_parser("test", help="Benchmark on train/valid data")
    add_model_args(test_parser)
    test_parser.add_argument("--split", default="valid")
    test_parser.add_argument("--data-dir", default=str(DATA_DIR))
    test_parser.add_argument("--max-question", type=int, default=100)
    test_parser.set_defaults(func=cmd_test)

    data_parser = subparsers.add_parser("generate_dataset", help="Generate RFT-style CoT data")
    add_model_args(data_parser)
    data_parser.add_argument("--split", default="train")
    data_parser.add_argument("--data-dir", default=str(DATA_DIR))
    data_parser.add_argument("--output-json", default="data/rft_vllm.json")
    data_parser.add_argument("--max-samples", type=int, default=None)
    data_parser.add_argument("--oversample", type=int, default=10)
    data_parser.add_argument("--temperature", type=float, default=0.8)
    data_parser.add_argument("--max-tokens", type=int, default=75)
    data_parser.set_defaults(func=cmd_generate_dataset)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
