# ========================================
# Imports
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch

import nathans_ml_utils.filesys as filesys

# ========================================
# Callbacks


class BackupCallback(pl.Callback):
    """Callback to backup checkpoints and logs every n epochs."""

    def __init__(self, backup_dir: Path | str, n_epochs: int = 5) -> None:
        self.n_epochs = n_epochs
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Backup every n_epochs
        if (trainer.current_epoch + 1) % self.n_epochs == 0:
            """Create backup of checkpoints and logs"""
            filesys.sync_dirs(
                f"{pl_module.config.run_path}/checkpoints",
                f"{pl_module.config.run_path}/backup/checkpoints",
            )
            filesys.sync_dirs(
                f"{pl_module.config.run_path}/logs",
                f"{pl_module.config.run_path}/backup/logs",
            )
            print("\ncheckpoints and logs backed up")


class ResourceLogger(pl.Callback):
    """Callback to log certain training resource stats"""

    def __init__(self, log_interval: int = 1):
        # Parameters
        self.cuda: bool = torch.cuda.is_available()
        self.log_interval: int = log_interval

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Mark initial values
        self.step_start_time = time.time()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Get time elapsed for one batch
        time_elapsed: float = time.time() - self.step_start_time

        # Get vram usage for batch
        if self.cuda:
            peak_mem: float = torch.cuda.max_memory_allocated(pl_module.device) / 1e9

        # Log
        if batch_idx % self.log_interval == 0:
            metrics: dict = {
                "step": trainer.global_step,
                "step_time_s": time_elapsed,
                "step_vram_gb": peak_mem,
            }
            trainer.logger.log_metrics(metrics, step=trainer.global_step)


class SanitizeCheckpointCallback(pl.Callback):
    """
    Lightning callback to automatically convert dict_keys, dict_values, and dict_items
    into lists before saving checkpoints, avoiding pickling errors.
    """

    def _sanitize(self, obj) -> Any:
        """Recursively sanitize objects to make them pickleable."""
        dict_values_type = type({}.values())
        dict_keys_type = type({}.keys())
        dict_items_type = type({}.items())

        if isinstance(obj, dict):
            return {self._sanitize(k): self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (dict_values_type, dict_keys_type, dict_items_type)):
            return list(obj)
        elif isinstance(obj, (list, tuple, set)):
            # Recursively sanitize container elements
            return type(obj)(self._sanitize(x) for x in obj)
        else:
            return obj

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict
    ) -> dict:
        """Hook called automatically before saving a checkpoint."""
        for attr_name in dir(pl_module):
            if attr_name.startswith("_"):
                continue
            # Skip read-only / internal attributes
            if hasattr(type(pl_module), attr_name) and isinstance(
                getattr(type(pl_module), attr_name), property
            ):
                continue

            attr_value = getattr(pl_module, attr_name)
            sanitized = self._sanitize(attr_value)

            setattr(pl_module, attr_name, sanitized)

        return checkpoint
