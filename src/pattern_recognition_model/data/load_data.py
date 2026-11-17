import json
import os

from pathlib import Path

# Resolves to .../src/pattern_recognition_model/data on Windows, Mac, and Linux
DATA_DIR = Path(__file__).resolve().parent
ROWS_PATH = os.path.join(DATA_DIR , "v1" , "data" , "rows.jsonl")
SEMANTIC_CACHE_DIR = os.path.join(DATA_DIR , "v1" , "data" , "semantic_cache")

def load_dataset(file: str | Path = ROWS_PATH) -> list[dict]:
    """
    Loads the dataset from a JSONL file and returns it as a list of dictionaries.
    """
    with Path(file).open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

load_dataset(file=ROWS_PATH)