"""
"patchwise poetry"

PatchGPT: Autoregressive Image Generation from Real Images
Training on local folder: ../images/iris/

Comments:
With depth of 1 the results are not good, but with depth of 2 the results are much better.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize

# Initialize device for MPS
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)
np.random.seed(42)
torch.mps.manual_seed(0)
torch.mps.set_per_process_memory_fraction(0.5)

NUM_EPOCHS = 3_000
SAVE_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 2e-3
PATCH_SIZE = 8
IMG_SIZE = 128
DEPTH = 2
N_HEADS = 8
N_DIMS = 128


def patchify(img, patch_size):
    """Patchify an image into smaller patches."""
    B, C, _, _ = img.shape
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)
    return patches.reshape(B, -1, C * patch_size * patch_size)


def unpatchify(patches, patch_size, img_size):
    """Unpatchify the patches back into an image."""
    B, N, D = patches.shape
    C = D // (patch_size**2)
    patches = patches.view(B, N, C, patch_size, patch_size)
    num = img_size // patch_size
    patches = patches.view(B, num, num, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.contiguous().view(B, C, img_size, img_size)


class ImageFolderDataset(Dataset):
    """Custom dataset for loading images from a folder."""

    def __init__(self, root, transform=None):
        self.paths = sorted(
            [str(f) for f in Path(root).iterdir() if f.suffix.lower() in (".jpg", ".png")]
        )
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

    def __len__(self):
        return len(self.paths)


def load_images(path, img_size=32, batch_size=4):
    """Load images from a folder and apply transformations."""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolderDataset(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class PatchGPT(nn.Module):
    """A simple PatchGPT model using PyTorch."""

    def __init__(self, input_dim, seq_len, dim=N_DIMS, heads=N_HEADS, depth=DEPTH):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))

        self.transformer_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=dim, nhead=heads) for _ in range(depth)]
        )

        # # add GELU activation
        self.activation = nn.GELU()
        for block in self.transformer_blocks:
            block.activation = self.activation

        self.output_proj = nn.Linear(dim, input_dim)

    def forward(self, x):
        """Forward pass through the model."""
        x = self.token_embed(x) + self.pos_embed[:, : x.size(1), :]
        for block in self.transformer_blocks:
            x = block(x)
            x = self.activation(x)
        return self.output_proj(x)


def train():
    """Train the PatchGPT model on images."""
    input_dim = 3 * PATCH_SIZE * PATCH_SIZE
    seq_len = (IMG_SIZE // PATCH_SIZE) ** 2

    model = PatchGPT(input_dim=input_dim, seq_len=seq_len).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()

    dataloader = load_images("../images/iris", img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            img, _ = batch
            img = img.to(DEVICE)

            patches = patchify(img, PATCH_SIZE)

            input_seq = patches[:, :-1]
            target_seq = patches[:, 1:]

            output = model(input_seq)
            loss = loss_fn(output, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % SAVE_EPOCHS == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
            generate_image(model, PATCH_SIZE, IMG_SIZE, seq_len, input_dim, DEVICE, epoch)


# Generation
@torch.no_grad()
def generate_image(model, patch_size, img_size, seq_len, input_dim, device, epoch):
    """Generate an image using the trained model."""
    model.eval()
    # Use real patch to seed generation
    dataloader = load_images("../images/iris", img_size=img_size, batch_size=1)
    real_img, _ = next(iter(dataloader))
    real_img = real_img.to(device)

    start = patchify(real_img, patch_size)[:, :1, :]  # First real patch only
    generated = [start]

    for _ in range(seq_len - 1):
        inp = torch.cat(generated, dim=1)
        out = model(inp)
        next_patch = out[:, -1:, :]
        generated.append(next_patch)

    all_patches = torch.cat(generated, dim=1)
    result = unpatchify(all_patches, patch_size, img_size).cpu().squeeze().permute(1, 2, 0)
    # plt.imshow(result)

    result_img = result.permute(2, 0, 1).unsqueeze(0)  # CHW
    upsampled = resize(result_img, size=[256, 256], interpolation=Image.Resampling.BICUBIC)
    plt.imshow(upsampled.squeeze().permute(1, 2, 0))

    plt.title(f"Generated Image @ Epoch {epoch}")
    Path("../images/outputs_iris").mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.savefig(f"../images/outputs_iris/gen_{epoch:03d}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    train()
