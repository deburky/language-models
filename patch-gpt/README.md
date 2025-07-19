# PatchGPT: Patchwise Autoregressive Image Generation

Author: https://github.com/deburky

## Abstract

We explore a GPT-style Vision Transformer that autoregressively generates images one patch at a time. Unlike diffusion or GAN-based image generation methods, this model predicts the next image patch based purely on prior patches in a sequence, using causal self-attention in a decoder-only Transformer. We propose a minimal prototype that demonstrates this mechanism and opens avenues for autoregressive visual modeling.

---

## 1. Introduction

Image generation is typically tackled with GANs or diffusion models, but recent research has revisited autoregressive approaches. Inspired by GPT, we propose a simplified decoder-only Vision Transformer (ViT) that generates images one patch at a time.

This paper presents a simple prototype of such a model using PyTorch and walks through the steps needed to build it.

---

## 2. Patchify and Unpatchify Functions

```python
import torch

def patchify(img, patch_size):
    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)  # (B, N_patches, C, P, P)
    return patches.reshape(B, -1, C * patch_size * patch_size)

def unpatchify(patches, patch_size, img_size):
    B, N, D = patches.shape
    C = D // (patch_size ** 2)
    patches = patches.view(B, N, C, patch_size, patch_size)
    num_patches_per_row = img_size // patch_size
    patches = patches.view(B, num_patches_per_row, num_patches_per_row, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.contiguous().view(B, C, img_size, img_size)
```

---

## 3. GPT-style Transformer Decoder for Vision

```python
import torch.nn as nn

class PatchGPT(nn.Module):
    def __init__(self, input_dim, seq_len, dim=512, heads=8, depth=6):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads), num_layers=depth
        )
        self.output_proj = nn.Linear(dim, input_dim)

    def forward(self, x):
        x = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output_proj(x)
```

---

## 4. Training Loop

```python
def train(model, dataloader, optimizer, criterion):
    model.train()
    for img, _ in dataloader:
        img = img.to(device)
        patches = patchify(img, patch_size=4)
        input_seq = patches[:, :-1]
        target_seq = patches[:, 1:]

        out = model(input_seq)
        loss = criterion(out, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 5. Autoregressive Generation

```python
@torch.no_grad()
def generate(model, start_patch, seq_len):
    model.eval()
    generated = [start_patch]

    for _ in range(seq_len - 1):
        inp = torch.cat(generated, dim=1)
        out = model(inp)
        next_patch = out[:, -1:, :]
        generated.append(next_patch)

    return torch.cat(generated, dim=1)
```

---

## 6. Conclusion

This minimal prototype demonstrates that GPT-style architectures can be extended to vision in a patchwise autoregressive way. Despite being inefficient for high-res images, this approach offers a clear, interpretable generative process and encourages future research into lightweight, autoregressive image generation.

---

## 7. Usage Instructions

### Prerequisites

First, install the dependencies using `uv` (recommended) or `pip`:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Available Scripts

The `tools/` directory contains several training and visualization scripts:

#### 1. Basic Training Scripts

**Train on Iris Dataset:**
```bash
python tools/train_patch_gpt_iris.py
```
This script trains PatchGPT on the iris flower images located in `images/iris/`. It uses predefined hyperparameters optimized for the iris dataset.

**Train on Checkerboard Patterns:**
```bash
python tools/train_patch_gpt_checkers.py
```
This script demonstrates training on synthetic checkerboard patterns. It's useful for understanding the basic autoregressive behavior of the model.

#### 2. CLI-based Training (with customizable parameters)

**General Training Script:**
```bash
python tools/patch_gpt.py train --help
```

Example usage with custom parameters:
```bash
python tools/patch_gpt.py train \
  --data-path images/iris \
  --num-epochs 3000 \
  --batch-size 4 \
  --learning-rate 2e-3 \
  --patch-size 8 \
  --img-size 128 \
  --depth 2 \
  --n-heads 8 \
  --n-dims 128
```

**Root-level CLI (Alternative):**
```bash
python patchgpt_cli.py train --help
```

Or use the provided shell script:
```bash
bash patch_gpt_cli.sh
```

#### 3. GIF Generation Scripts (Training with Animation)

These scripts create animated GIFs showing the patch-by-patch generation process:

**Iris Dataset with GIF Output:**
```bash
python tools/patch_gpt_gif_iris.py train \
  --data-path images/iris \
  --num-epochs 3000 \
  --save-epochs 100
```

**Pattern Dataset with GIF Output:**
```bash
python tools/patch_gpt_gif_patterns.py train \
  --data-path images/patterns \
  --num-epochs 2000 \
  --patch-size 4 \
  --img-size 32
```

**Checkerboard with GIF Output:**
```bash
python tools/patch_gpt_gif_checkers.py
```

#### 4. Utility Scripts

**Generate Checkerboard Patterns:**
```bash
python tools/generate_checkerboard.py
```
This script generates and visualizes synthetic checkerboard patterns.

### CLI Parameters

All CLI scripts support the following parameters:

- `--data-path`: Path to training images folder (default varies by script)
- `--num-epochs`: Number of training epochs (default: 3000-5000)
- `--save-epochs`: Save outputs every N epochs (default: 100)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 2e-3)
- `--patch-size`: Size of image patches (default: 8)
- `--img-size`: Input image resolution (default: 128)
- `--depth`: Transformer depth/layers (default: 2)
- `--n-heads`: Number of attention heads (default: 8)
- `--n-dims`: Transformer embedding dimension (default: 128)

### Output

During training, the scripts will:
- Print loss values every `save_epochs` iterations
- Generate sample images showing training progress
- For GIF scripts: Create animated visualizations of the generation process

The generated images and GIFs are saved in the same directory as the training script, typically named with epoch numbers and timestamps.

### Dataset Structure

Your image datasets should be organized as follows:
```
images/
├── iris/           # Iris flower images
├── patterns/       # Pattern images  
└── your_dataset/   # Your custom images
```

Images should be in common formats (JPG, PNG) and will be automatically resized to the specified `img_size`.