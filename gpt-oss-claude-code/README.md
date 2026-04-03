# Testing

```bash
make serve MODEL=model-rank32-q4
```

# 1. Back up existing adapters

```bash
cp -r adapters adapters-rank32-backup
```

# 2. Train with num_layers: -1

```bash
make finetune
```

```bash
cp adapters/0000150_adapters.safetensors adapters/adapters.safetensors
```

# 3. Fuse new adapters to a new path

```bash
uv run python -m mlx_lm fuse \
    --model mlx-community/gpt-oss-20b-MXFP4-Q4 \
    --adapter-path adapters \
    --save-path model-rank32-v2-fused --dequantize
```

# 4. Quantize

```bash
uv run python -m mlx_lm convert \
    --hf-path model-rank32-v2-fused \
    --mlx-path gpt-oss-claude-mlx \
    --quantize --q-bits 4 --q-group-size 64
```

# 5. Serve

```bash
make serve MODEL=gpt-oss-claude-mlx
make chat MODEL=gpt-oss-claude-mlx
```

# 6. Upload to Hugging Face

```bash
cd gpt-oss-claude-mlx && export HF_TOKEN=$(grep HF_WRITE_TOKEN /Users/deburky/Documents/python-ml-projects/.env | cut -d'"' -f2) && hf upload deburky/gpt-oss-claude-mlx .
```