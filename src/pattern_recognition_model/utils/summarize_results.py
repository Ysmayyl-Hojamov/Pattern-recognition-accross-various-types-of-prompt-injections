import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mlp_random_seeds_output")
case_name = sys.argv[2] if len(sys.argv) > 2 else folder.name

PATTERNS = {
    "auc": r"ROC AUC:\s*([0-9.]+)",
    "recall": r"RECALL:\s*([0-9.]+)%",
    "benign_fpr": r"Benign FPR:\s*([0-9.]+)%",
    "auth_fpr": r"Authorized FPR:\s*([0-9.]+)%",
    "rec_fpr5": r"FPR<=5%:\s*recall=([0-9.]+)",
    "rec_fpr10": r"FPR<=10%:\s*recall=([0-9.]+)",
    "rec_fpr20": r"FPR<=20%:\s*recall=([0-9.]+)",
}

grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
    lambda: defaultdict(list)
)

for path in sorted(folder.glob("*.txt")):
    text = path.read_text()
    held_match = re.search(r"Held out:\s*(.+)", text)
    if not held_match:
        continue
    split_name = held_match.group(1).strip()

    spliced_match = re.search(r"Is spliced:\s*(\w+)", text)
    is_spliced = spliced_match.group(1).strip() if spliced_match else "N/A"

    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            grouped[(split_name, is_spliced)][key].append(float(m.group(1)))


def fmt(vals: list[float], is_pct: bool = False) -> str:
    if not vals:
        return "N/A"
    suffix = "%" if is_pct else ""
    return f"{np.mean(vals):.2f}{suffix} ± {np.std(vals):.2f}{suffix}"


lines = [
    "| Held Out | Spliced | Seeds | ROC AUC | Recall @ 0.5 | Recall @ FPR≤10% | Benign FPR | Authorized FPR |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
]

for (split_name, is_spliced), metrics in sorted(grouped.items()):
    n = len(metrics["auc"])
    lines.append(
        f"| `{split_name}` | {is_spliced} | {n} "
        f"| {fmt(metrics['auc'])} "
        f"| {fmt(metrics['recall'], True)} "
        f"| {fmt(metrics['rec_fpr10'])} "
        f"| {fmt(metrics['benign_fpr'], True)} "
        f"| {fmt(metrics['auth_fpr'], True)} |"
    )

report = "\n".join(lines) + "\n"

out_dir = Path("mlp_random_seeds_output") / "summarized_results"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"{case_name}_summary.txt"
out_file.write_text(report)

print(report)
print(f"Saved summary to {out_file}")
