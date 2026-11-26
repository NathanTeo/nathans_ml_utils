"""
Author: Nathan Teo

Description: General utility functions, added on from the original EDM codebase.
"""

"""Imports"""
import psutil
import torch

"""Utils"""


def show_ram(
    prefix="",
) -> tuple[float, None | float, None | float] | tuple[float, None, None]:
    # RAM usage
    ram: float = psutil.Process().memory_info().rss / 1024**2  # in MB

    # VRAM usage (if CUDA is available)
    if torch.cuda.is_available():
        vram: float = torch.cuda.memory_allocated() / 1024**2  # in MB
        vram_reserved: float = torch.cuda.memory_reserved() / 1024**2  # in MB
        print(
            f"{prefix} RAM: {ram:.1f} MB | VRAM allocated: {vram:.1f} MB | VRAM reserved: {vram_reserved:.1f} MB"
        )
        return ram, vram, vram_reserved
    else:
        print(f"{prefix} RAM: {ram:.1f} MB (no GPU)")
        return ram, None, None
