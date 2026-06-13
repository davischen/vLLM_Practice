# Chapter 2 Results and Explanations

This file records command outputs from `ch2_llm_serving_examples.py` and explains
which Chapter 2 concepts each result demonstrates.

When adding a new result, use this pattern:

```text
Command:
python3 -B ch2_llm_serving_examples.py --section <section>

Key output:
...

Explanation:
...
```

## Result 1: Tokenizer

Recorded: 2026-06-13 11:42:38 CST

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section tokenizer
```

Key output:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%
tokenizer_config.json: 100%
vocab.json: 100%
merges.txt: 100%
tokenizer.json: 100%

Text: Write a short introduction about the US capital city.
Token count: 10
00 token=         'Write' id=7985
01 token=            'Ġa' id=264
02 token=        'Ġshort' id=2805
03 token= 'Ġintroduction' id=16800
04 token=        'Ġabout' id=911
05 token=          'Ġthe' id=279
06 token=           'ĠUS' id=2274
07 token=      'Ġcapital' id=6722
08 token=         'Ġcity' id=3283
09 token=             '.' id=13
```

Chapter concept:

Tokenizer and embedding input preparation. Chapter 2 explains that an LLM does
not consume raw text directly. The tokenizer first converts text into token
pieces, then maps each token to a token ID. The model later maps those IDs into
embedding vectors.

Explanation:

The Hugging Face download lines are expected on the first run. This section uses
`Qwen/Qwen2.5-0.5B`, so the tokenizer files are fetched from the Hugging Face Hub:
`config.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, and
`tokenizer.json`. This is not the full model weight download.

The warning means the request is unauthenticated. The run still works, but
setting `HF_TOKEN` can provide higher rate limits and faster downloads.

The prompt:

```text
Write a short introduction about the US capital city.
```

was split into 10 tokens. Each token has a vocabulary ID:

- `Write` maps to token ID `7985`.
- `Ġa`, `Ġshort`, `Ġabout`, and similar tokens include the `Ġ` marker, which
  indicates that the token begins after a whitespace character.
- `.` is its own token with ID `13`.

This output demonstrates why token count can differ from word count. The model
does not see a sentence as a sentence; it sees a sequence of integer token IDs.

What to try next:

```bash
python3 -B ch2_llm_serving_examples.py --section config
python3 -B ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 5
```

## Result 2: Model Configuration

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section config
```

Key output:

```text
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors: 100%|...| 988M/988M [00:55<00:00, 17.7MB/s]
Loading weights: 100%|...| 290/290 [00:01<00:00, 164.71it/s]
generation_config.json: 100%|...| 138/138 [00:00<00:00, 2.00MB/s]

=== Model Configuration Parameters ===

Architecture Parameters:
Hidden size: 896
Number of layers: 24
Number of attention heads: 14
Intermediate size: 4864

Tokenizer Parameters:
Vocabulary size: 151936
Maximum position embeddings: 32768

Model Size:
Total parameters: 494,032,768
Trainable parameters: 494,032,768

Selected Model-specific Parameters:
dtype: float16
is_encoder_decoder: False
vocab_size: 151936
hidden_size: 896
intermediate_size: 4864
num_hidden_layers: 24
num_attention_heads: 14
num_key_value_heads: 2
hidden_act: silu
max_position_embeddings: 32768
rms_norm_eps: 1e-06
use_cache: True
tie_word_embeddings: True
rope_parameters: {'rope_theta': 1000000.0, 'rope_type': 'default'}
attention_dropout: 0.0
bos_token_id: 151643
eos_token_id: 151643
_name_or_path: Qwen/Qwen2.5-0.5B
output_attentions: False
```

Chapter concept:

Decoder-only Transformer architecture inspection. Chapter 2 recommends checking
the model configuration before serving, because these parameters determine memory
usage, compute cost, context length, KV-cache size, and serving strategy.

Explanation:

This run downloaded the actual model weights:

```text
model.safetensors: 988M
```

Unlike the tokenizer result, this section loads the model itself. The download is
therefore much larger and takes longer. The `torch_dtype` deprecation warning is
from the Transformers library API. It does not break the run; it means newer
Transformers versions prefer `dtype` over `torch_dtype`.

Important architecture values:

- `hidden_size: 896` means each token is represented internally as a 896-dimensional hidden vector.
- `num_hidden_layers: 24` means the model has 24 decoder blocks.
- `num_attention_heads: 14` means each attention layer has 14 query heads.
- `num_key_value_heads: 2` means the model uses grouped-query attention (GQA): many query heads share fewer key/value heads. This reduces KV-cache memory compared with full multi-head attention.
- `intermediate_size: 4864` is the feed-forward/MLP expansion size inside each decoder block.
- `vocab_size: 151936` is the number of possible output token IDs.
- `max_position_embeddings: 32768` is the configured maximum context length.
- `use_cache: True` means the model supports KV-cache reuse during generation.
- `dtype: float16` means weights are loaded in half precision in this environment, reducing memory compared with float32.

Serving interpretation:

The model has `494,032,768` parameters, so it is roughly a 0.5B parameter model.
This is small enough for local experimentation, but still large enough to show
real LLM serving behaviors such as tokenizer loading, model weight loading,
decoder-layer structure, and KV-cache generation.

The context length of `32768` tokens is large, but using long prompts increases
prefill cost and KV-cache memory. In serving systems, this matters because longer
contexts reduce how many concurrent requests can fit in memory.

The `use_cache: True` setting connects directly to the next Chapter 2 concept:
manual token generation with KV cache. With caching enabled, the model can reuse
previous key/value tensors instead of recomputing all earlier tokens on every
decode step.

What to try next:

```bash
python3 -B ch2_llm_serving_examples.py --section decoder
python3 -B ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 5
```

## Result 3: Setup Section

Command:

```bash
python3 -B ch2_llm_serving_examples.py
```

Key output:

```text
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
```

Chapter concept:

Chapter 2 separates serving internals into small runnable pieces: tokenizer,
model configuration, decoder structure, attention, pipeline generation, manual
decode, KV cache, and vLLM serving examples.

Explanation:

The default section is intentionally lightweight. It does not load model weights
or import optional ML libraries, so it is a safe first command to run before
installing `torch`, `transformers`, or `vllm`.

## Result 4: vLLM Configuration Snippet

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section vllm-config
```

Key output:

```text
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
    stop=["\n", "###"],
    frequency_penalty=0.1,
    presence_penalty=0.1,
    repetition_penalty=1.1,
    skip_special_tokens=True,
)
```

Chapter concept:

vLLM turns the lower-level serving ideas from Chapter 2 into practical serving
knobs: prefix cache, chunked prefill, KV-cache block size, maximum context
length, active sequence limit, and sampling parameters.

Explanation:

This section prints a configuration example only; it does not instantiate vLLM.
That makes it runnable on macOS or CPU-only machines. To run actual vLLM
inference, use a Linux/CUDA environment and the `vllm-basic`, `vllm-batch`, or
`vllm-stream` sections.

## Result 5: Missing Dependency Error

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section tokenizer
```

Possible output before installing dependencies:

```text
RuntimeError: This section needs PyTorch and Hugging Face Transformers.
Install them with `pip install torch transformers accelerate`.
```

Chapter concept:

Chapter 2 examples have two layers: local configuration examples and real model
execution examples. Real model execution needs PyTorch and Hugging Face
Transformers because the script loads a tokenizer or model from the Hugging Face
Hub.

Explanation:

If this error appears, install the required packages in the active virtual
environment, then rerun the section:

```bash
pip install torch transformers accelerate matplotlib
python3 -B ch2_llm_serving_examples.py --section tokenizer
```

Once dependencies are installed, the tokenizer section should produce token/id
rows like Result 1.
