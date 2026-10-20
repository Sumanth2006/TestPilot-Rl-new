"""
Data loader module for TestPilot-RL.
Loads, downloads, and processes real-world Continuous Integration (CI) datasets:
- ABB Paint Control (paintcontrol.csv)
- ABB IOF/ROL (iofrol.csv)
Originally introduced by Spieker et al. (ISSTA 2017) in the RETECS benchmark.
"""

import os
import ast
import urllib.request
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


PAINTCONTROL_URL = "https://raw.githubusercontent.com/mregorova/RETECS/main/DATA/paintcontrol.csv"
IOFROL_URL = "https://raw.githubusercontent.com/mregorova/RETECS/main/DATA/iofrol.csv"


def download_dataset(dataset_name: str, target_dir: str = "data") -> str:
    """
    Downloads the specified industrial dataset if not already present.
    """
    os.makedirs(target_dir, exist_ok=True)
    filename = f"{dataset_name}.csv"
    filepath = os.path.join(target_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return filepath

    url_map = {
        "paintcontrol": PAINTCONTROL_URL,
        "iofrol": IOFROL_URL,
    }

    if dataset_name not in url_map:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(url_map.keys())}")

    url = url_map[dataset_name]
    print(f"Downloading {dataset_name} dataset from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
        f.write(resp.read())

    print(f"Downloaded {dataset_name} ({os.path.getsize(filepath):,} bytes) -> {filepath}")
    return filepath


def load_ci_dataset(dataset_name: str = "paintcontrol", data_dir: str = "data") -> pd.DataFrame:
    """
    Loads and cleans a CI dataset into a pandas DataFrame.
    Semi-colon separated with columns:
    Id, Name, Duration, CalcPrio, LastRun, LastResults, Verdict, Cycle
    """
    filepath = download_dataset(dataset_name, data_dir)
    df = pd.read_csv(filepath, sep=";")

    # Ensure correct data types
    df["Name"] = df["Name"].astype(str)
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(1.0)
    # Ensure durations are positive
    df["Duration"] = df["Duration"].clip(lower=1.0)
    df["Verdict"] = pd.to_numeric(df["Verdict"], errors="coerce").fillna(0).astype(int)
    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce").fillna(0).astype(int)

    def parse_results(val):
        if isinstance(val, str):
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    pass
        return []

    df["LastResults"] = df["LastResults"].apply(parse_results)
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes summary statistics for a CI dataset.
    """
    total_executions = len(df)
    unique_tests = df["Name"].nunique()
    total_cycles = df["Cycle"].nunique()
    total_failures = (df["Verdict"] == 1).sum()
    failure_rate = (total_failures / total_executions) * 100 if total_executions > 0 else 0.0

    return {
        "total_executions": total_executions,
        "unique_test_cases": unique_tests,
        "total_ci_cycles": total_cycles,
        "total_failures": total_failures,
        "failure_rate_pct": failure_rate,
        "avg_tests_per_cycle": total_executions / total_cycles if total_cycles > 0 else 0,
        "avg_duration": float(df["Duration"].mean()),
    }


def split_ci_cycles(
    df: pd.DataFrame, train_ratio: float = 0.70
) -> Tuple[List[int], List[int]]:
    """
    Splits CI cycles chronologically into train and held-out test cycles.
    """
    cycles = sorted(df["Cycle"].unique())
    split_idx = int(len(cycles) * train_ratio)
    train_cycles = cycles[:split_idx]
    test_cycles = cycles[split_idx:]
    return train_cycles, test_cycles
