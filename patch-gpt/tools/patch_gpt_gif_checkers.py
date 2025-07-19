"""
PatchGPT + Patchwise Animation: Watch a Transformer hallucinate a checkerboard.
"""

import io
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
if DEVICE.type == "mps":
    torch.mps.manual_seed(0)
    torch.mps.set_per_process_memory_fraction(0.5)

# Hyperparameters
IMG_SIZE = 32
PATCH_SIZE = 2
INPUT_DIM = PATCH_SIZE**2
SEQ_LEN = (IMG_SIZE // PATCH_SIZE) ** 2
EPOCHS = 200
SAVE_INTERVAL = 20
DIM = 128
HEADS = 4
DEPTH = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


def generate_checkerboard(batch_size=16, img_size=32):
    """Generate a checkerboard pattern."""
    img = torch.zeros((batch_size, 1, img_size, img_size))
    for b in range(batch_size):
        for i in range(img_size):
            for j in range(img_size):
                img[b, 0, i, j] = (i + j) % 2
    return img


def patchify(img, patch_size):
    """Convert an image into patches."""
    B, C, _, _ = img.shape
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    return patches.reshape(B, -1, C * patch_size * patch_size)


def unpatchify(patches, patch_size, img_size):
    """Convert patches back into an image."""
    B, N, D = patches.shape
    C = D // (patch_size**2)
    patches = patches.view(B, N, C, patch_size, patch_size)
    num = img_size // patch_size
    patches = patches.view(B, num, num, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.contiguous().view(B, C, img_size, img_size)


class PatchGPT(nn.Module):
    """A simple Transformer-based patch autoregressive model."""

    def __init__(self, input_dim, seq_len, dim=128, heads=4, depth=4):
        """Initialize the PatchGPT model."""
        super().__init__()
        self.token_embed = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(dim, input_dim)

    def forward(self, x):
        """Forward pass through the model."""
        x = self.token_embed(x) + self.pos_embed[:, : x.size(1), :]
        x = self.transformer(x)
        return self.output_proj(x)


def train():
    """Train the PatchGPT model."""
    model = PatchGPT(input_dim=INPUT_DIM, seq_len=SEQ_LEN, dim=DIM, heads=HEADS, depth=DEPTH).to(
        DEVICE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        img = generate_checkerboard(batch_size=BATCH_SIZE, img_size=IMG_SIZE).to(DEVICE)
        patches = patchify(img, PATCH_SIZE)

        input_seq = patches[:, :-1]
        target_seq = patches[:, 1:]

        output = model(input_seq)
        loss = loss_fn(output, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % SAVE_INTERVAL == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    generate_gif(model)


@torch.no_grad()
def generate_gif(model):
    """Generate a GIF of the model's predictions."""
    model.eval()
    Path("../images/outputs_checkers").mkdir(parents=True, exist_ok=True)

    start = patchify(generate_checkerboard(1), patch_size=PATCH_SIZE)[:, :1, :].to(DEVICE)
    generated = [start]

    gif_frames = []

    for step in range(SEQ_LEN - 1):
        inp = torch.cat(generated, dim=1)
        out = model(inp)
        next_patch = out[:, -1:, :]
        generated.append(next_patch)

        current = torch.cat(generated, dim=1)
        missing = SEQ_LEN - current.shape[1]
        if missing > 0:
            pad = torch.zeros((1, missing, INPUT_DIM), device=DEVICE)
            current = torch.cat([current, pad], dim=1)

        image = unpatchify(current, PATCH_SIZE, IMG_SIZE).cpu().squeeze()

        fig, ax = plt.subplots(facecolor="white")
        ax.imshow(image, cmap="gray_r")
        ax.axis("off")
        ax.set_title(f"Patch Step {step + 1}")
        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0.5, dpi=150, facecolor="white"
        )
        plt.close(fig)
        buf.seek(0)
        gif_frames.append(imageio.v2.imread(buf))

    gif_path = "../images/outputs_checkers/generated_checkerboard.gif"
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, gif_frames, duration=0.15)
    print(f"Saved animated patch generation to: {gif_path}")


if __name__ == "__main__":
    train()
