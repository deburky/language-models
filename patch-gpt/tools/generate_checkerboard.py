import torch
from matplotlib import pyplot as plt


def generate_checkerboard(batch_size=16, img_size=32):
    """Generate a checkerboard pattern."""
    img = torch.zeros((batch_size, 1, img_size, img_size))
    for b in range(batch_size):
        for i in range(img_size):
            for j in range(img_size):
                img[b, 0, i, j] = (i + j) % 2
    return img


# Generate checkerboard images
generate_checkerboard(batch_size=1, img_size=32)
plt.imshow(generate_checkerboard(batch_size=1, img_size=32)[0, 0].cpu(), cmap="gray")
