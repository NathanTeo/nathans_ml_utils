# ========================================
# Imports

import csv
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

# ========================================
# Loggers

class ResumableCSVLogger(Logger):
    def __init__(
        self,
        save_dir: str | Path,
    ):
        super().__init__()

        self._save_dir = Path(save_dir)

        self._log_dir: Path = self._save_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_file: Path = self._log_dir / "metrics.csv"
        self._hparams_file: Path = self._log_dir / "hparams.yaml"

        self._metrics_keys = []

        if self._metrics_file.exists():
            with self._metrics_file.open(newline="") as f:
                reader: csv.DictReader[str] = csv.DictReader(f)
                self._metrics_keys = reader.fieldnames or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def save_dir(self) -> str:
        return str(self._save_dir)

    @property
    def log_dir(self) -> str:
        return str(self._log_dir)

    @rank_zero_only
    def log_hyperparams(self, params: dict | Namespace) -> None:
        if isinstance(params, Namespace):
            params = vars(params)

        with self._hparams_file.open("w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

    @rank_zero_only
    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        row = {
            key: value.item() if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }

        if step is not None:
            row["step"] = step

        # Identify newly introduced columns
        new_keys: set[str] = set(row) - set(self._metrics_keys)

        if new_keys:
            self._metrics_keys: list[str] = sorted(set(self._metrics_keys) | new_keys)

            # Rewrite existing rows if the CSV schema changes
            if self._metrics_file.exists():
                with self._metrics_file.open(newline="") as f:
                    existing_rows: list[dict] = list(csv.DictReader(f))

                with self._metrics_file.open("w", newline="") as f:
                    writer: csv.DictWriter[str] = csv.DictWriter(
                        f,
                        fieldnames=self._metrics_keys,
                    )
                    writer.writeheader()
                    writer.writerows(existing_rows)

        # Append the new row
        file_exists: bool = self._metrics_file.exists()

        with self._metrics_file.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._metrics_keys,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    @rank_zero_only
    def save(self) -> None:
        pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        pass
