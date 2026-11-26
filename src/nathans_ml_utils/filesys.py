# ========================================
# Imports
import shutil
from pathlib import Path

# ========================================
# Functions


def mkdirs(paths: list[Path] | list[str]) -> None:
    """Create directories if they do not exist.

    Args:
        paths: list of paths to create.
    """
    # Ensure path type
    paths = [Path(path) for path in paths]

    # Create dirs
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clr_paths(paths: list[Path] | list[str]) -> None:
    """Deletes all directories or files in list

    Args:
        paths: list of paths to delete
    """
    # Ensure path type
    paths = [Path(path) for path in paths]

    # Delete file/dir
    for path in paths:
        if path.exists():
            (path.unlink() if path.is_file() else shutil.rmtree(path))


def sync_dirs(src: Path, dst: Path):
    """Sync all directory contents between src and dst.

    Args:
        src (Path): Source path.
        dst (Path): Destination path.
    """
    # Enforcing pathlib
    src = Path(src)
    dst = Path(dst)

    # Ensure destination exists
    dst.mkdir(parents=True, exist_ok=True)

    # Copy/update all files and directories from src to dst
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            sync_dirs(item, target)  # recursive
        else:
            shutil.copy2(item, target)

    # Delete files and dirs in dst that don't exist in src
    for item in list(dst.iterdir()):
        if not (src / item.name).exists():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
