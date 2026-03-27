#!/usr/bin/env python3
"""
Shared utility functions for AMELA pipeline scripts.
Consolidates common patterns across generate, synthesize, etc.

Import and use functions directly:
    from utils import load_manifest_rows, round_robin, print_device_info
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import torch


# ==========================================
# Manifest I/O
# ==========================================


def load_manifest_rows(manifest_path: str) -> list[dict]:
    """
    Load manifest rows from CSV or JSONL file.

    Args:
        manifest_path: Path to .csv or .jsonl file

    Returns:
        List of dict entries
    """
    path = Path(manifest_path)

    if path.suffix == ".csv":
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

    elif path.suffix in [".jsonl", ".json"]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f]

    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")


def save_manifest(entries: list[dict], manifest_path: str):
    """
    Save manifest to CSV or JSONL file.

    Args:
        entries: List of dict entries
        manifest_path: Path to .csv or .jsonl file
    """
    path = Path(manifest_path)

    if path.suffix in [".jsonl", ".json"]:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    elif path.suffix == ".csv":
        if not entries:
            return

        # Collect all unique field names
        fieldnames = []
        for entry in entries:
            for key in entry.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")



# ==========================================
# Task Distribution
# ==========================================


def round_robin(items: list, task_id: int, num_tasks: int) -> list:
    """Distribute items across parallel tasks using round-robin."""
    return [item for i, item in enumerate(items) if i % num_tasks == task_id]


# ==========================================
# Timestamp Utilities
# ==========================================


def timestamp_now(fmt: str = "full") -> str:
    """
    Get formatted timestamp.

    Args:
        fmt: Format type - 'full', 'time', 'date', 'iso'
    """
    formats = {
        "full": "%Y-%m-%d %H:%M:%S",
        "time": "%H:%M:%S",
        "date": "%d-%m-%y",
    }

    if fmt == "iso":
        return datetime.now().isoformat()
    return datetime.now().strftime(formats.get(fmt, formats["full"]))


# ==========================================
# Device/CUDA Setup
# ==========================================


def print_device_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("Device: CPU")
