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