# ========================================
# Imports
import time
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from pandas import DataFrame
import torch
import json
import shutil
from datetime import timedelta
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import nathans_ml_utils.filesys as filesys

# ========================================
# Configured callbacks

# DRAC progress bar configuration
drac_progress_bar = TQDMProgressBar(refresh_rate=20, leave=False)


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



class TrainingSummaryCallback(L.Callback):
    def __init__(
        self,
        output_dir: Path | str,
        metrics_path: Optional[Path | str] = None,
    ):
        super().__init__()

        self.output_dir = Path(output_dir)
        self.metrics_path = Path(metrics_path) if metrics_path else None

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve metrics path
        metrics_path = self.metrics_path

        if metrics_path is None:
            metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"

        if not metrics_path.exists():
            return

        df: DataFrame = pd.read_csv(metrics_path)

        self.save_training_summary(df, pl_module)
        self.plot_logged_data(df)


    def save_training_summary(
        self,
        df: pd.DataFrame,
        pl_module: L.LightningModule,
    ) -> None:
        """Save JSON of training summary statistics."""

        def get_stat(df, col, func):
            if col not in df.columns:
                return None

            return func(df[col].to_numpy())

        summary: dict = {
            "steps": get_stat(df, "step", np.nanmax),
            "epochs": get_stat(df, "epoch", np.nanmax),
            "total_training_time": get_stat(
                df,
                "step_time_s",
                lambda x: str(timedelta(seconds=np.nansum(x))),
            ),
            "mean_step_time_s": get_stat(df, "step_time_s", np.nanmean),
            "min_loss": get_stat(df, "train_loss", np.nanmin),
            "peak_vram_gb": get_stat(df, "step_vram_gb", np.nanmax),
        }

        summary: dict = {k: v for k, v in summary.items() if v is not None}

        summary["learnable_params"] = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )

        with (self.output_dir / "training_summary_stats.json").open("w") as f:
            json.dump(summary, f, indent=2)

    def plot_logged_data(self, df: pd.DataFrame) -> None:
        """Plot all logged metrics."""

        columns = df.columns[~df.columns.isin(["epoch", "step"])]
        n_plots = len(columns)

        if n_plots == 0:
            return

        fig, axs = plt.subplots(
            n_plots,
            1,
            figsize=(5, 3 * n_plots),
            squeeze=False,
        )

        for ax, column in zip(axs.flat, columns):
            ax.plot(df["step"], df[column])
            ax.set_xlabel("Step")
            ax.set_ylabel(column)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "logged_metrics.png")
        plt.close()