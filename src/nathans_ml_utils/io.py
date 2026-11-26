# ========================================
# Imports
import importlib
from pathlib import Path
from typing import Any

import pandas as pd

# ========================================
# Functions


def load_config(path: str | Path) -> dict:
    """Loads a config file.

    Args:
        path (str): path of config file.

    Returns:
        module: config file.
    """
    path = Path(path)

    # Create module
    spec = importlib.util.spec_from_file_location("config", path)

    # Execute module
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module


def add_row_csv(path: str | Path, entry: dict, id: Any = None) -> None:
    """Function to update a saved csv with a new entry.

    Args:
        path (str): Path of csv.
        entry (dict): New entries.
        id (str, optional): Column name of unique identifier. If None, first column is used. Defaults to None.
    """
    path = Path(path)

    # Load or create df
    if path.exists(path):
        df: pd.DataFrame = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    # Ensure all keys exist as columns
    for key in entry.keys():
        if key not in df.columns:
            df[key] = None  # Add missing columns

    # Removes entries that share the same unique identifier
    if id is None:  # Assume first column is id
        first_col = df.columns[0]
        df = df[df[first_col] != entry[first_col]]
    else:
        df = df[df[id] != entry[id]]

    # Convert entry to df and append
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    # Save updated csv
    df.to_csv(path, index=False)


def append_csv(src: Path, dst: Path, id_columns: list[str]) -> None:
    """Append csv to another csv, used to append logs. If destination does not exist, append to empty table.
    Tables should share the same columns.

    Args:
        src (Path): Source csv path.
        dst (Path): Destination csv path.
        id_columns (list): List of columns that uniquely identify a row.
    """
    # Enforcing pathlib
    src = Path(src)
    dst = Path(dst)

    # Get new entry
    entry: pd.DataFrame = pd.read_csv(src)
    entry = entry.groupby(id_columns).first().reset_index()  # Avoid duplicates

    # Get destination csv
    if dst.exists():
        df: pd.DataFrame = pd.read_csv(dst)
    else:
        df = pd.DataFrame()

    # Combine dfs
    df = pd.concat([df, entry], ignore_index=True)

    # Save df
    df.to_csv(dst, index=False)
