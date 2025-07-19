"""
This script trains a simple PatchGPT model on a synthetic dataset.
It is a functioning autoregressive image model from scratch that learns coherent structure.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

# Initialize device for MPS
device = torch.device("mps")
torch.mps.manual_seed(0)
torch.mps.set_per_process_memory_fraction(0.5)


# Simple synthetic dataset: checkerboard
def generate_checkerboard(batch_size=16, img_size=32):
    """Generate a checkerboard pattern."""
    img = torch.zeros((batch_size, 1, img_size, img_size))
    for b in range(batch_size):
        for i in range(img_size):
            for j in range(img_size):
                img[b, 0, i, j] = (i + j) % 2
    return img


def patchify(img, patch_size):
    """Convert an image into patches.
    B, C, H, W = batch size, channels, height, width
    """
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
    """A simple PatchGPT model using PyTorch."""

    def __init__(self, input_dim, seq_len, dim=128, heads=4, depth=4):
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


def train():  # sourcery skip: extract-method
    """Trains the PatchGPT model on a synthetic dataset."""
    patch_size = 4
    img_size = 32
    seq_len = (img_size // patch_size) ** 2
    input_dim = patch_size**2

    model = PatchGPT(input_dim=input_dim, seq_len=seq_len, dim=128, heads=4, depth=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        img = generate_checkerboard(batch_size=16, img_size=img_size).to(device)
        patches = patchify(img, patch_size)

        # Autoregressive: predict next patch from previous ones
        # IMPORTANT HERE
        #  patches[:, 0] is the first patch, and we want to predict patches[:, 1]
        input_seq = patches[:, :-1]
        target_seq = patches[:, 1:]

        output = model(input_seq)
        loss = loss_fn(output, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # Test generation
    with torch.no_grad():
        # start = torch.zeros((1, 1, input_dim), device=device)
        start = patchify(generate_checkerboard(1), patch_size=4)[:, :1, :]  # Real patch first
        generated = [start]
        for _ in range(seq_len - 1):
            inp = torch.cat(generated, dim=1)
            out = model(inp)
            next_patch = out[:, -1:, :]
            generated.append(next_patch)

        all_patches = torch.cat(generated, dim=1)
        result = unpatchify(all_patches, patch_size, img_size).cpu().squeeze()

        plt.imshow(result, cmap="gray")
        plt.title("Generated Pattern")
        plt.savefig("../images/generated_pattern.png")
        print("Done")


if __name__ == "__main__":
    train()
