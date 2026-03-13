# llama.fs

**LLM inference in F#**

A from-scratch port of the [Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) inference engine written in F#, running on .NET 10 with [TorchSharp](https://github.com/dotnet/TorchSharp). Supports CUDA for GPU acceleration.

---

## What is it?

`llama.fs` lets you run a Llama language model locally from a .NET 10 F# application — no Python, no Ollama, no external runtime. It loads the original Meta checkpoint (`.pth` file) directly and performs autoregressive text generation using the same architecture as the reference Python implementation.

The interactive loop lets you chat with the model from the terminal, using the Llama 3 instruct template so the model responds as an assistant.

---

## How it works

The project is structured in five files:

| File | Responsibility |
|---|---|
| `Utils.fs` | RoPE frequency precomputation (`precomputeThetaPosFrequencies`), rotary embedding application, KV-head repeat |
| `Tokenizer.fs` | Tiktoken BPE tokenizer, special token handling for the instruct template |
| `Model.fs` | Full Transformer architecture: `RMSNorm`, `SelfAttention` (GQA), `FeedForward` (SwiGLU), `TransformerBlock`, `Transformer` |
| `Llama.fs` | Model loading via `TorchSharp.PyBridge`, KV-cache generation loop, top-p sampling |
| `Program.fs` | Interactive CLI loop with instruct template formatting |

### Architecture highlights

- **Grouped Query Attention (GQA)** — `n_kv_heads=8` with `n_heads=32`, KV heads are repeated 4× via `repeatKV`
- **Rotary Positional Embeddings (RoPE)** — computed with `torch.polar` and applied as complex multiplication
- **SwiGLU activation** — feedforward uses three weight matrices (`w1`, `w2`, `w3`)
- **RMSNorm** — normalization before attention and feedforward blocks
- **BFloat16** — weights loaded in BF16 for memory efficiency
- **KV cache** — past keys/values are cached across generation steps using `narrow(...).copy_()` views

### Generation

Text is generated autoregressively: at each step, the model produces logits for the next token. If `temperature > 0`, top-p (nucleus) sampling is used; otherwise greedy argmax. Generation stops when the EOS token (`<|eot_id|>`, id `128009`) is produced or `maxGenLen` is reached.

---

## Model: Llama 3.2 1B Instruct

### Which model

[`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) — Meta's 1-billion parameter instruction-tuned model.

```
dim            = 2048
n_layers       = 16
n_heads        = 32
n_kv_heads     = 8
vocab_size     = 128 256
rope_theta     = 500 000
ffn_multiplier = 1.5×
dtype          = BFloat16
```

### Why Llama 3.2 1B Instruct?

| Reason | Detail |
|---|---|
| **GPU constraint** | The development machine has a 6 GB VRAM GPU (RTX 4050), so 1B is the largest model that fits |
| **Instruct-tuned** | Responds as an assistant out of the box with the Llama 3 chat template |
| **Open weights** | Downloadable from Meta / HuggingFace with a free license |
| **Standard checkpoint** | Ships as a single `.pth` file loadable directly by TorchSharp.PyBridge |
| **Good baseline** | Small enough to iterate fast, capable enough to give meaningful answers |

The architecture implementation is model-agnostic — any Llama 3.x checkpoint with a compatible `params.json` should work. If you have more VRAM, larger models should run with no code changes beyond pointing `modelFolder` at the new checkpoint.

### Other models to try

Below is a list of Llama models from the same family. The implementation should be compatible with all of them (same architecture, larger weights). Contributions are welcome — see [Contributing](#contributing).

| Model | VRAM (approx.) | Notes |
|---|---|---|
| `Llama-3.2-1B-Instruct` ✅ | ~3 GB | Tested — this repo |
| `Llama-3.2-3B-Instruct` | ~7 GB | Next step up, still small GPU friendly |
| `Llama-3.1-8B-Instruct` | ~16 GB | Strong general-purpose model |
| `Llama-3.1-8B` | ~16 GB | Base (non-instruct) variant |
| `Llama-3.3-70B-Instruct` | ~140 GB | Requires multi-GPU or CPU offload |
| `Llama-3.1-405B-Instruct` | ~800 GB | Research / datacenter scale |

---

## Contributing

If you test this with a larger model, please open a Pull Request including:
- Which model you used (name + parameter count)
- Your hardware (GPU model + VRAM)
- Any code changes needed (if any)
- A sample prompt/response showing it works

---

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- NVIDIA GPU with CUDA 12.1+ (or CPU, slower)
- Llama model weights from the [meta-llama/llama-models](https://github.com/meta-llama/llama-models) repository
  - `consolidated.00.pth`
  - `params.json`
  - `tokenizer.model`

## Running

1. Download the model weights from [meta-llama/llama-models](https://github.com/meta-llama/llama-models) and place them in a local folder, e.g. `C:\Models\Llama3.2-1B-Instruct`
2. Update the `modelFolder` path in `Program.fs`:

```fsharp
let modelFolder = @"C:\Models\Llama3.2-1B-Instruct"
```

3. Run:

```bash
dotnet run
```

The first run will download NuGet packages (~2 GB for `libtorch-cuda-12.1-win-x64`). Loading the checkpoint takes ~25 seconds on first run.

```
Using device: cuda
Loading checkpoint took 24521 ms
Model loaded. Type your prompt and press Enter (or 'exit' to quit).

You: _
```

---

## Sample outputs

### Logic reasoning

```
You: If A > B and B > C, which one is the greatest?
Assistant: If A > B and B > C, then A is the greatest.

You can think of it as a chain: A is greater than B, and B is greater than C,
so A is at the top of the chain and is therefore the greatest.
```

### Code generation

```
You: Write a Python function that reverses a string.
Assistant: Here is a simple Python function that reverses a string:

def reverse_string(s):
    return s[::-1]

This function uses Python's slice notation with a step of -1 to reverse the string.
For example:

print(reverse_string("hello"))  # Output: "olleh"
print(reverse_string("world"))  # Output: "dlrow"
```

### Opinion

```
You: Do you like F#?
Assistant: F# is a functional-first programming language developed by Microsoft,
and it has a lot of features that many developers appreciate. Here are some
aspects of F# that are often praised:

- Concise syntax: F# code is often more concise than equivalent C# code,
  which can make it easier to read and write.
- Strong type system: F# has a strong, statically-typed type system that can
  help catch errors at compile time.
- Functional programming: F# is designed around functional programming
  principles, which can lead to more predictable and composable code.

Whether you personally "like" F# depends on your programming background,
preferences, and the type of projects you work on. If you enjoy functional
programming or are looking for a language with a strong type system, F# might
be a great fit for you!
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| [TorchSharp](https://github.com/dotnet/TorchSharp) | 0.101.5 | Tensor operations, neural network modules |
| `libtorch-cuda-12.1-win-x64` | 2.1.0.1 | Native LibTorch binaries with CUDA 12.1 |
| `TorchSharp.PyBridge` | 1.2.0 | Load PyTorch `.pth` checkpoints directly |
| [Microsoft.ML.Tokenizers](https://github.com/dotnet/machinelearning) | 0.21.1 | Tiktoken BPE tokenizer base |

---

## Project structure

```
llama.fs/
├── llama.fs.fsproj   # Project file (.NET 10, F# 9 preview)
├── Utils.fs          # RoPE utilities
├── Tokenizer.fs      # Tiktoken BPE tokenizer
├── Model.fs          # Transformer architecture
├── Llama.fs          # Model loading and generation
└── Program.fs        # Interactive CLI entry point
```

---

## Inspiration & References

This project was inspired by and based on:

- **[hkproj/pytorch-llama](https://github.com/hkproj/pytorch-llama)** — LLaMA implemented from scratch in PyTorch by Umar Jamil. The primary reference for the architecture and generation logic.
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — The de-facto standard for efficient LLM inference in C/C++. Inspiration for running LLMs natively without Python.
- **[Microsoft.ML.GenAI.LLaMA](https://github.com/dotnet/machinelearning/tree/main/src/Microsoft.ML.GenAI.LLaMA)** 
— Official LLaMA implementation in the ML.NET repository, also built on TorchSharp.

### Learning resources

- 📹 [Coding LLaMA 2 from scratch in PyTorch — Umar Jamil](https://www.youtube.com/watch?v=oM4VmoabDAI) — Step-by-step walkthrough of the LLaMA architecture: KV cache, GQA, RoPE, SwiGLU.
- 📹 [But what is a GPT? Visual intro to transformers — 3Blue1Brown](https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=7&t=1400s) — Visual explanation of transformers and attention from the Neural Networks series.