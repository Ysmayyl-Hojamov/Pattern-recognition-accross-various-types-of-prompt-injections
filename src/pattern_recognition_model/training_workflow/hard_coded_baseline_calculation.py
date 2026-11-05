import json
from os import fpathconf
from pathlib import Path

from pattern_recognition_model.training_workflow.logreg_baseline import DANGEROUS_TOOLS

dataset = Path(__file__).parent / "v1" / "rows.jsonl"

with open(dataset, "r", encoding="utf-8") as file:
    """
    Getting recall. benign FPR. Authorized FPR.
    Benign FPR is cases when the run was benign but the detector marked it as a 1.
    """
    data = [json.loads(line) for line in file if line.strip()]
    total = 0

    # Detecting accross the whole dataset
    true_positive = 0
    false_positive = 0

    true_negative = 0
    false_negative = 0

    # Authorized rows.
    authorized_tp = 0
    authorized_fp = 0

    authorized_tn = 0
    authorized_fn = 0

    # Benign rows.
    benign_tp = 0
    benign_fp = 0

    benign_tn = 0
    benign_fn = 0

    for row in data:
        r_tool = row["tool"]
        row_label = row["label"]
        row_kind = row["kind"]

        # remove spliced runs
        if not row["trace_id"].startswith("runs/"):
            continue

        total += 1

        if r_tool in DANGEROUS_TOOLS and row_label == 1:
            """
            Attack was there and the rule
            """
            true_positive += 1
        elif r_tool in DANGEROUS_TOOLS and row_label == 0:
            """
            The rule deemed it as an attack while it wasn't
            """
            false_positive += 1

        elif not r_tool in DANGEROUS_TOOLS and row_label == 0:
            """
            Was not an attack and detector passed it.
            """
            true_negative += 1
        elif not r_tool in DANGEROUS_TOOLS and row_label == 1:
            """
            It was an actual attack, but the detector passed it
            """
            false_negative += 1

        # Authorized calls
        if r_tool in DANGEROUS_TOOLS and row_label == 1 and row_kind == "authorized":
            authorized_tp += 1
        elif r_tool in DANGEROUS_TOOLS and row_label == 0 and row_kind == "authorized":
            authorized_fp += 1

        if (
            not r_tool in DANGEROUS_TOOLS
            and row_label == 1
            and row_kind == "authorized"
        ):
            authorized_fn += 1
        if (
            not r_tool in DANGEROUS_TOOLS
            and row_label == 0
            and row_kind == "authorized"
        ):
            authorized_tn += 1

        # Benign calls
        if r_tool in DANGEROUS_TOOLS and row_label == 1 and row_kind == "benign":
            benign_tp += 1
        elif r_tool in DANGEROUS_TOOLS and row_label == 0 and row_kind == "benign":
            benign_fp += 1

        if not r_tool in DANGEROUS_TOOLS and row_label == 1 and row_kind == "benign":
            benign_fn += 1
        if not r_tool in DANGEROUS_TOOLS and row_label == 0 and row_kind == "benign":
            benign_tn += 1

    print(f"TOTAL: {total}")
    print(f"RECALL = {true_positive/(true_positive+false_negative) * 100: .2f}%")
    print(f"Benign FPR = {benign_fp/(benign_fp+benign_tn) * 100: .2f}%")
    print(f"Authorized FPR = {authorized_fp/(authorized_fp+authorized_tn) * 100: .2f}%")
