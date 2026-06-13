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

## Performance and Cost Summary

| Area | Baseline / observation | Optimized / comparison | Performance or cost lesson |
| --- | --- | --- | --- |
| Tokenization | Default prompt becomes 10 tokens | Token IDs are the unit the model actually serves | Prompt length directly affects latency, context use, and API token cost. |
| Model size | `Qwen/Qwen2.5-0.5B` has 494,032,768 parameters | `float16` is used on GPU/MPS when available | Weight dtype and parameter count determine memory cost before any KV cache is allocated. |
| Manual generation | No KV cache: 1.3218s total, 0.0658s/token | KV cache: 0.3690s total, 0.0182s/token | KV cache is about 3.58x faster in this local run because decode reuses prior keys and values. |
| Prefill vs decode | KV prefill latency: 0.0511s | Average KV decode latency: 0.0165s/token | Serving cost has two phases: prompt prefill cost and per-token decode cost. |

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

## Result 6: Decoder Layer Structure

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section decoder
```

Key output:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|...| 290/290 [00:00<00:00, 1719.46it/s]

Model top-level structure:
model: Qwen2Model
  embed_tokens: Embedding
  layers: ModuleList
  norm: Qwen2RMSNorm
  rotary_emb: Qwen2RotaryEmbedding
lm_head: Linear

Decoder layer 0 structure:
self_attn: Qwen2Attention
  attention details:
    head_dim: 64
  q_proj: Linear
  k_proj: Linear
  v_proj: Linear
  o_proj: Linear
mlp: Qwen2MLP
  gate_proj: Linear
  up_proj: Linear
  down_proj: Linear
  act_fn: SiLUActivation
input_layernorm: Qwen2RMSNorm
post_attention_layernorm: Qwen2RMSNorm
```

Chapter concept:

Decoder-only Transformer block anatomy. Chapter 2 describes how token
representations pass through repeated decoder layers made of self-attention,
MLP/feed-forward projections, rotary position embeddings, and normalization.

Explanation:

The top-level model has an embedding table, a stack of decoder layers, a final
RMSNorm, rotary embeddings, and an `lm_head` that projects hidden states back to
vocabulary logits. Inside decoder layer 0, `q_proj`, `k_proj`, `v_proj`, and
`o_proj` are the attention projections, while `gate_proj`, `up_proj`, and
`down_proj` form the MLP block.

## Result 7: Attention Visualization

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section attention
```

Key output:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[transformers] The following generation flags are not valid and may be ignored: ['output_attentions'].
Loading weights: 100%|...| 290/290 [00:00<00:00, 2678.89it/s]
<IPython.core.display.HTML object>
<IPython.core.display.HTML object>
<IPython.core.display.Javascript object>
```

Chapter concept:

Attention visualization. Chapter 2 uses attention inspection to show that each
generated token can attend to previous prompt tokens with different weights
across layers and heads.

Explanation:

Because `bertviz` is installed in this environment, the section created
IPython/HTML visualization objects instead of printing the textual fallback.
In a notebook or VS Code interactive window, those objects render as an
attention view. In a normal terminal, Python only prints object summaries such
as `<IPython.core.display.HTML object>`.

## Result 8: Manual Generation Without KV Cache

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section manual-no-cache --max-new-tokens 20
```

Key output:

```text
000 next token: ' We' (0.7624s)
001 next token: ' are' (0.0668s)
002 next token: ' looking' (0.0895s)
003 next token: ' for' (0.0539s)
004 next token: ' published' (0.0537s)
...
019 next token: ' Living' (0.0630s)

Total time: 2.3933s
Average time/token: 0.1194s
```

Chapter concept:

Autoregressive decoding without cache. Each new token is generated by feeding
the full growing sequence back through the model.

Explanation:

This section demonstrates the simplest generation loop: run the model, sample
one next token, append it to the input IDs, and repeat. Without KV cache, every
step recomputes attention over all previous tokens, which becomes increasingly
expensive as the sequence grows.

## Result 9: Manual Generation With KV Cache

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section manual-kv-cache --max-new-tokens 20
```

Key output:

```text
000 prefill input_shape=(1, 55) next=' This' (0.1315s)
001 decode  input_shape=(1, 1) next=' book' (0.6138s)
002 decode  input_shape=(1, 1) next=' examines' (0.0161s)
003 decode  input_shape=(1, 1) next='\n' (0.0171s)
...
019 decode  input_shape=(1, 1) next=' our' (0.0163s)

Total time: 1.0602s
Average time/token: 0.0528s
Prefill latency: 0.1315s
Average decode latency: 0.0486s
```

Chapter concept:

Prefill/decode split and KV-cache reuse. The model first processes the whole
prompt once, then each later decode step only needs the latest token plus cached
keys and values from earlier tokens.

Explanation:

The first row is the prefill phase with `input_shape=(1, 55)`, meaning the whole
prompt is processed. Later rows use `input_shape=(1, 1)`, meaning the model only
receives the newly generated token while reusing `past_key_values`. This is the
core serving optimization behind fast token-by-token generation.

## Result 10: No-cache vs KV-cache Comparison

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section compare-manual --max-new-tokens 20
```

Key output:

```text
=== Without KV cache ===
000 next token: ' The' (0.1357s)
001 next token: ' ' (0.0514s)
...
019 next token: 'IR' (0.0597s)

Total time: 1.3218s
Average time/token: 0.0658s

=== With KV cache ===
000 prefill input_shape=(1, 55) next=' How' (0.0511s)
001 decode  input_shape=(1, 1) next=' might' (0.0358s)
002 decode  input_shape=(1, 1) next=' our' (0.0157s)
...
019 decode  input_shape=(1, 1) next=' are' (0.0154s)

Total time: 0.3690s
Average time/token: 0.0182s
Prefill latency: 0.0511s
Average decode latency: 0.0165s
Matplotlib is building the font cache; this may take a moment.
```

Chapter concept:

KV cache changes the cost profile of generation. Instead of recomputing the
whole sequence for every token, decode steps reuse cached key/value tensors and
operate on one new token at a time.

Explanation:

In this recorded run, the no-cache loop took `1.3218s` for 20 tokens, averaging
`0.0658s/token`. The KV-cache loop took `0.3690s`, averaging `0.0182s/token`.
That is about a `3.58x` speedup for this short local run:

```text
1.3218 / 0.3690 = 3.58x
```

The exact numbers vary by hardware and generated tokens, but the serving lesson
is stable: KV cache makes the decode phase much cheaper and is essential for
interactive LLM serving.

## Result 11: Cached Model Configuration Re-run

Command:

```bash
python3 -B ch2_llm_serving_examples.py --section config
```

Key output:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|...| 290/290 [00:01<00:00, 240.75it/s]

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
transformers_version: 5.12.0
dtype: float16
num_key_value_heads: 2
use_cache: True
rope_parameters: {'rope_theta': 1000000.0, 'rope_type': 'default'}
_name_or_path: Qwen/Qwen2.5-0.5B
```

Chapter concept:

Model configuration inspection is repeatable and should be part of serving
debugging. The same model architecture values are visible even after the model
files are already cached locally.

Explanation:

Compared with the first `config` run in Result 2, this run no longer shows the
large `model.safetensors` download because the weights were already cached. It
still loads the same 0.5B model and confirms the serving-relevant settings:
24 layers, 14 attention heads, 2 KV heads, 32k context length, and `use_cache:
True`.
