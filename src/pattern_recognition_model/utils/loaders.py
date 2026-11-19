"""
Everything connected to datasets and data.
"""

import json
import os
from pathlib import Path
from typing import Any
from copy import deepcopy

import torch

BASE_DIR = Path(__file__).resolve().parent.parent


def load_dataset(dataset_version, include_spliced=False) -> list[Any]:
    with open(
        os.path.join(BASE_DIR, "data", dataset_version, "data", "rows.jsonl"),
        "r",
        encoding="utf-8",
    ) as file:
        data = [json.loads(line) for line in file if line.strip()]

        if not include_spliced:
            for row in deepcopy(data):
                if row["trace_id"].startswith("runs_spliced/"):
                    data.remove(row)

    return data


def load_emb(embedding_type, dataset_v: str = "v1"):
    return torch.load(
        os.path.join(BASE_DIR, "data", dataset_v, f"emb_{embedding_type}.pt")
    )
