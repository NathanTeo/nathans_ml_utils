# ========================================
# Imports
from typing import Literal, Optional

import matplotlib as mpl
import torch

# ----------------------------------------------------------------------------
# Plotting functions


def plot_image_grid(
    images: torch.Tensor,
    axes: mpl.axes.Axes,
    titles: Optional[list[str]] = None,
    normalize: Optional[Literal["image", "batch"]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = "viridis",
) -> None:
    """Plot images on axes. Assunes images are in CPU.

    Args:
        images: Tensor of shape (B,C,H,W)
        axes: Matplotlib axes object or array of axes objects
        titles: List of titles for each image
        normalize: Whether to normalize images to [0, 1] before plotting
    """
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else axes

    # Checks
    if images.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor (B,C,H,W), got shape {tuple(images.shape)}"
        )
    if flat_axes.size < len(images):
        raise ValueError(
            f"Number of axes ({flat_axes.size}) less than number of images ({len(images)})"
        )
    # Normalize images to [0, 1]
    if normalize == "batch":
        images: torch.Tensor = (images - images.min()) / (images.max() - images.min())

    # Plot each image
    for idx in range(len(flat_axes)):
        if idx < len(images):
            # CHW → HWC
            img: torch.Tensor = images[idx].permute(1, 2, 0)
            # Normalize image to [0, 1]
            if normalize == "image":
                img: torch.Tensor = (img - img.min()) / (img.max() - img.min())
            # Plot
            flat_axes[idx].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            # Create title
            if titles is not None:
                flat_axes[idx].set_title(titles[idx], fontsize=8)
        flat_axes[idx].axis("off")
