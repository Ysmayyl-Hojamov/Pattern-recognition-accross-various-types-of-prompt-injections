"""
Everything connected to datasets and data.
"""

import json
import os
from pathlib import Path
from typing import Any
from copy import deepcopy

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
