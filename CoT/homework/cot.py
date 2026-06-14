from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Convert units. Be concise. Show brief reasoning. Put the final number in <answer> tags.",
            },
            {
                "role": "user",
                "content": "How many MB are there per 9 GB?",
            },
            {
                "role": "assistant",
                "content": "1 GB = 1000 MB. 9 * 1000 = 9000.0.\n<answer>9000.0</answer>",
            },
            {
                "role": "user",
                "content": "How do we express 4 pound in terms of ounce?",
            },
            {
                "role": "assistant",
                "content": "1 pound = 16 ounces. 4 * 16 = 64.0.\n<answer>64.0</answer>",
            },
            {
                "role": "user",
                "content": "Can you change 9 kmh to its equivalent in mi/h",
            },
            {
                "role": "assistant",
                "content": "1 kmh = 0.6213711922373 mi/h. 9 * 0.6213711922373 = 5.592340730136007.\n<answer>5.592340730136007</answer>",
            },
            {
                "role": "user",
                "content": "What is the equivalent of 10 century in year?",
            },
            {
                "role": "assistant",
                "content": "1 century = 100 years. 10 * 100 = 1000.0.\n<answer>1000.0</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
