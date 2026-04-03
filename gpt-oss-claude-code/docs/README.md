# claude-code-mlx

Fine-tune gpt-oss-20b on Anthropic-style responses, shrink it to 4-bit, and serve it
via Ollama with the harmony chat template baked in.

## Pipeline overview

```
Q8 base model
    ↓  LoRA fine-tune (style adaptation)
adapter weights (~50 MB)
    ↓  fuse
fine-tuned Q8 model (~21 GB)
    ↓  re-quantize to 4-bit
fine-tuned Q4 model (~10 GB)  ← final artifact
    ↓
Ollama (harmony template) → litellm → Claude Code /v1/messages
```

---

## Step 1 — Prepare the dataset

Converts `anthropic_style_finetune.jsonl` to `{"text": ...}` lines (full harmony-rendered
conversation) via the tokenizer chat template. The `thinking` field → `analysis` channel;
`content` → `final` channel. Alternate corpus: `fine_tuning.jsonl` (pass `--input`).

```shell
uv run python prepare_dataset.py
# → data/train.jsonl (24 examples), data/valid.jsonl (6 examples)
```

## Step 2 — LoRA fine-tune

Train on the Q8 base (better gradient quality than Q4).
72 iterations ≈ 3 epochs over 24 examples.

```shell
uv run --env-file ../.env python -m mlx_lm lora --model mlx-community/gpt-oss-20b-MXFP4-Q8 \
  --train --data ./data --batch-size 1 --num-layers 8 --iters 72 \
  --steps-per-eval 12 --save-every 36 \
  --adapter-path adapters
```

Watch validation loss — if it starts rising before 72 iters, stop early with Ctrl-C.
The best checkpoint is saved automatically at `adapters/`.

## Step 3 — Fuse + re-quantize to 4-bit

Fuse the adapter into the model, then shrink from Q8 (~21 GB) to Q4 (~10 GB):

```shell
uv run python -m mlx_lm fuse --model mlx-community/gpt-oss-20b-MXFP4-Q8 --adapter-path adapters --save-path model-finetuned-q8 --dequantize
```

```shell
uv run python -m mlx_lm convert --hf-path model-finetuned-q8 --mlx-path model-finetuned-q4 --quantize --q-bits 4 --q-group-size 64
```

`--dequantize` during fuse produces clean bf16 weights so the re-quantization to Q4 is
applied fresh rather than compounding quantization errors from Q8→Q4.

Result: `model-finetuned-q4/` — fine-tuned, ~10 GB, Apple Silicon optimized.

## Step 4 — Serve via mlx-lm (OpenAI format)

```shell
uv run python -m mlx_lm server --model model-finetuned-q4 --port 8080
```

## Step 5 — Serve via Ollama (with harmony template baked in)

Convert to GGUF, then import with the Modelfile:

```shell
# Install llama.cpp if not already installed
brew install llama.cpp

# Convert fused model to GGUF Q4_K_M
llama-gguf-split --merge model-finetuned-q8 model-finetuned.gguf  # if needed
llama-quantize model-finetuned-q8/model.gguf model-finetuned-q4.gguf Q4_K_M

# Or: convert HF safetensors directly
python3 $(brew --prefix llama.cpp)/convert_hf_to_gguf.py model-finetuned-q8 \
  --outfile model-finetuned.gguf --outtype q4_K_M
```

Update the Modelfile `FROM` line and recreate the Ollama model:

```shell
# Edit Modelfile: change FROM line to point to the local GGUF
ollama create gpt-oss-20b -f Modelfile
ollama run gpt-oss-20b
```

## Step 6 — Claude Code (/v1/messages)

Ollama exposes OpenAI format; Claude Code needs Anthropic format.
Bridge with litellm (single command, no custom server):

```shell
pip install litellm
litellm --model ollama/gpt-oss-20b --port 8082
```

```shell
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=local
claude --model gpt-oss-20b
```

Or skip Ollama and use the MLX model directly (faster on Apple Silicon):

```shell
uv run python -m mlx_lm server --model model-finetuned-q4 --port 8080
litellm --model openai/gpt-oss-20b --api_base http://localhost:8080 --port 8082
```

---

## About the harmony format

gpt-oss was trained with a structured token format:

| Token         | Role                            |
|---------------|---------------------------------|
| `<|start|>`   | begins a message block          |
| `<|message|>` | separates role from content     |
| `<|channel|>` | precedes channel name           |
| `<|end|>`     | ends a completed turn           |
| `<|return|>`  | end of generation (stop token)  |

The model outputs a chain-of-thought in the `analysis` channel before the final answer
in the `final` channel. This is expected behavior.

---

## Quick reference — size vs quality

| Model                              | Size   | RAM    | Notes                    |
|------------------------------------|--------|--------|--------------------------|
| Base Q8 (training base)            | ~21 GB | ~23 GB | best quality, too large  |
| Fine-tuned Q4 (this pipeline)      | ~10 GB | ~12 GB | fine-tuned + small       |
| `unsloth/gpt-oss-20b-GGUF:Q4_K_M` | ~12 GB | ~14 GB | base only, no fine-tune  |
| Metal weights (original)           | 13 GB  | ~14 GB | MXFP4, custom backend    |
