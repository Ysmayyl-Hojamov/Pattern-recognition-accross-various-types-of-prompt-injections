"""
Combining arg_from_read and hard-coded baseline. Results in yet another baseline to compare against.
A smarter approach. However, here I use the hard-coded list of tools, which makes it not that veritable. It's just another
baseline to compare against.

My results:
RECALL:  55.80% (from 59.10% -> 55.80%)
Benign FPR:  4.41% (from 6.13% to 4.41%)
Authorized FPR:  12.00% (from 42.29% to 12.00%)
"""

from cProfile import label
import json
import os
from pathlib import Path

from pattern_recognition_model.training_workflow.logreg_baseline import DANGEROUS_TOOLS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "data", "v1", "data", "rows.jsonl")

with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file if line.strip()]

    kinds = set()

    # Detecting accross the whole dataset
    tp = fn = 0
    # Authorized rows.
    auth_fp = auth_tn = 0

    # For benign kind.
    benign_fp = benign_tn = 0

    for row in data:

        if not row["trace_id"].startswith("runs/"):
            continue

        r_label = int(row["label"])
        r_kind = row["kind"]
        is_tool_dangerous = row["tool"] in DANGEROUS_TOOLS

        # flagged.
        r_only_from_read = row["arg_only_from_read"]

        if r_label == 1:
            if r_only_from_read and is_tool_dangerous:
                tp += 1
            else:
                fn += 1

        # benign rows
        if r_kind == "benign" and r_label == 0:
            if r_only_from_read and is_tool_dangerous:
                benign_fp += 1

            else:
                benign_tn += 1

        # authorized rows
        if r_kind == "authorized" and r_label == 0:
            if r_only_from_read and is_tool_dangerous:
                auth_fp += 1
            else:
                auth_tn += 1

    recall = tp / (tp + fn)
    benign_fpr = benign_fp / (benign_fp + benign_tn)
    auth_fpr = auth_fp / (auth_fp + auth_tn)

    print(f"RECALL: {recall* 100: .2f}%")
    print(f"Benign FPR: {benign_fpr * 100: .2f}%")
    print(f"Authorized FPR: {auth_fpr* 100: .2f}%")
