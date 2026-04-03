# SETUP.md

LoRA → fuse → Q4 → Claude Code via local `/v1/messages`. Details: [README.md](README.md).

Run `uv` from the project root (`claude-code-mlx`). Optional: `../.env` with `HF_TOKEN` if the hub needs it.

## File layout

```
claude-code-mlx/
├── Makefile
├── sh/
│   └── serve.sh          # starts mlx_lm + bridge
├── scripts/
│   ├── bridge.py         # Anthropic /v1/messages → mlx_lm bridge
│   ├── download.py       # download base model from HF
│   ├── prepare_dataset.py
│   └── test_model.py
├── data/                 # train.jsonl + valid.jsonl
├── adapters/             # LoRA checkpoints
├── model-finetuned-q8/   # fused bf16 model
└── model-finetuned-q4/   # final quantized model (~10 GB)
```

## MLX-LM

[mlx-lm](https://github.com/ml-explore/mlx-lm) helps you run LLMs locally on Apple Silicon.

```shell
python -m mlx_lm lora      # LoRA fine-tuning
python -m mlx_lm fuse      # merge adapters into base model
python -m mlx_lm convert   # quantize (fp16 → Q4/Q8)
python -m mlx_lm server    # OpenAI-compatible inference server
```

## Training pipeline

```shell
make prepare     # convert anthropic_style_finetune.jsonl → data/
make finetune    # LoRA fine-tune on Q8 base (best checkpoint: iter 36)
make fuse        # fuse adapters + dequantize to bf16
make quantize    # re-quantize to Q4 (~10 GB)
```

Or run steps manually:

```shell
uv run python scripts/prepare_dataset.py

uv run --env-file ../.env python -m mlx_lm lora \
  --model mlx-community/gpt-oss-20b-MXFP4-Q8 \
  --train --data ./data --batch-size 1 --num-layers 8 --iters 72 \
  --steps-per-eval 12 --save-every 36 \
  --adapter-path adapters

cp adapters/0000036_adapters.safetensors adapters/adapters.safetensors

uv run --env-file ../.env python -m mlx_lm fuse \
  --model mlx-community/gpt-oss-20b-MXFP4-Q8 \
  --adapter-path adapters \
  --save-path model-finetuned-q8 --dequantize

uv run python -m mlx_lm convert \
  --hf-path model-finetuned-q8 \
  --mlx-path model-finetuned-q4 \
  --quantize --q-bits 4 --q-group-size 64
```

## Serving

```shell
make serve          # starts mlx_lm (:8080) + bridge (:8082)
make test-model     # run test prompts directly against the model
```

Override defaults:

```shell
make serve MODEL=model-finetuned-q4 PROXY_PORT=8082
```

## Smoke tests

```shell
make curl-test      # non-streaming /v1/messages
make curl-stream    # streaming /v1/messages
```

## Launch Claude Code

```shell
make claude
```

Or manually:

```shell
mkdir -p /tmp/claude-local && HOME=/tmp/claude-local \
ANTHROPIC_BASE_URL=http://localhost:8082 \
ANTHROPIC_API_KEY=local \
claude --model gpt-oss-20b
```
