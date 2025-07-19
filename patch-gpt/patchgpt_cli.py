"""patch_gpt.py: A simple implementation of a transformer model for image generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, resize

app = typer.Typer()

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)
torch.manual_seed(42)
np.random.seed(42)
if DEVICE.type == "mps":
    torch.mps.manual_seed(0)
    torch.mps.set_per_process_memory_fraction(0.5)


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
    """Implementation of a simple image dataset loader."""

    def __init__(self, root, transform=None):
        self.paths = sorted(
            [str(f) for f in Path(root).iterdir() if f.suffix.lower() in (".jpg", ".png")]
        )
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.paths)


def load_images(path, img_size, batch_size):
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
    """A simple implementation of a transformer model for image generation."""

    def __init__(self, input_dim, seq_len, dim, heads, depth):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))
        self.transformer_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=dim, nhead=heads) for _ in range(depth)]
        )
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


def generate_image(model, patch_size, img_size, seq_len, input_dim, device, epoch, data_path):  # pylint: disable=unused-argument
    """Generate an image using the trained model."""
    model.eval()
    dataloader = load_images(data_path, img_size=img_size, batch_size=1)
    real_img, _ = next(iter(dataloader))
    real_img = real_img.to(device)

    start = patchify(real_img, patch_size)[:, :1, :]
    generated = [start]

    for _ in range(seq_len - 1):
        inp = torch.cat(generated, dim=1)
        out = model(inp)
        next_patch = out[:, -1:, :]
        generated.append(next_patch)

    all_patches = torch.cat(generated, dim=1)
    result = unpatchify(all_patches, patch_size, img_size).cpu().detach().squeeze().permute(1, 2, 0)
    result_img = result.permute(2, 0, 1).unsqueeze(0)
    upsampled = resize(result_img, size=[256, 256], interpolation=InterpolationMode.BICUBIC)
    plt.imshow(upsampled.squeeze().permute(1, 2, 0))

    plt.title(f"Generated Image @ Epoch {epoch}")
    Path("images/outputs").mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.savefig(f"images/outputs/gen_{epoch:03d}.png", dpi=150)
    plt.close()


@app.command()
def train(
    data_path: str = typer.Option("images/iris", help="Folder with training images"),
    num_epochs: int = typer.Option(5000, help="Total training epochs"),
    save_epochs: int = typer.Option(100, help="Save output every N epochs"),
    batch_size: int = typer.Option(4, help="Training batch size"),
    learning_rate: float = typer.Option(2e-3, help="Learning rate"),
    patch_size: int = typer.Option(8, help="Patch size"),
    img_size: int = typer.Option(128, help="Input image resolution"),
    depth: int = typer.Option(2, help="Transformer depth"),
    n_heads: int = typer.Option(8, help="Number of attention heads"),
    n_dims: int = typer.Option(128, help="Transformer embedding dimension"),
):
    """Train PatchGPT on images."""
    input_dim = 3 * patch_size * patch_size
    seq_len = (img_size // patch_size) ** 2

    model = PatchGPT(
        input_dim=input_dim, seq_len=seq_len, dim=n_dims, heads=n_heads, depth=depth
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    dataloader = load_images(data_path, img_size=img_size, batch_size=batch_size)

    for epoch in range(num_epochs):
        for batch in dataloader:
            img, _ = batch
            img = img.to(DEVICE)

            patches = patchify(img, patch_size)
            input_seq = patches[:, :-1]
            target_seq = patches[:, 1:]

            output = model(input_seq)
            loss = loss_fn(output, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % save_epochs == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
            generate_image(
                model, patch_size, img_size, seq_len, input_dim, DEVICE, epoch, data_path
            )


if __name__ == "__main__":
    app()
