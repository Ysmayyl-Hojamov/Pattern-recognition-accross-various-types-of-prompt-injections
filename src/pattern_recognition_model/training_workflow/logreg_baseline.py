"""Logistic-regression baseline over agent behaviour rows.

What this is
------------
The simplest possible *learned* detector. One row is one tool call. The model
sees only what a monitor could see at decision time - the user's request, the
action, and what the agent had already read - and predicts whether this action
is the one an attacker wanted.

Why logistic regression before anything neural
----------------------------------------------
The data has ~7,700 rows but only 131 distinct user prompts, so rows are heavily
duplicated and a high-capacity model can memorise prompts instead of learning
behaviour. A linear model cannot memorise that way, trains in a second, and its
coefficients tell us which features it relies on - which the thesis explicitly
asks for.

Features (no leakage)
---------------------
Only columns a real monitor would have are used. eliberately excluded because
they encode the answer: `attack_type`, `trace_id`, `security`, `utility`,z
`kind`, `label_match`, `injection_task_id`, `injection_seen_before`, `pipeline`.
`read_digest` is excluded too: it is raw text for a later text model, not a
numeric feature.

Evaluation
----------
Two regimes, because they answer different questions:

  A. user-task-held-out   - random-ish split, but grouped so no user task
                            appears on both sides. Measures in-distribution skill.
  B. cross-suite          - train on workspace, test on banking/slack/travel.
                            Tool names are disjoint between suites, so the model
                            cannot fall back on "this tool is bad".

The operating threshold is chosen on BENIGN rows only, at a fixed false-positive
rate, and then applied to attacks. Choosing it on attack data would be cheating.

Reported together: recall, FPR on benign traffic, FPR on authorized requests
(the cases where the user legitimately asked for the same action). The last one
is the number that separates detecting hijacking from detecting dangerous tools.

Usage:
    uv run python -m pattern_recognition_model.training_workflow.logreg_baseline
    uv run python -m pattern_recognition_model.training_workflow.logreg_baseline --fpr 0.01
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that a monitor could legitimately observe at decision time.
NUMERIC_FEATURES = [
    "step",  # position of this action in the trace
    "n_calls",  # how long the whole episode was
    "n_str_args",  # string arguments in this call
    "frac_args_in_prompt",  # share of argument values the user actually said
    "frac_args_in_read",  # share that appeared in documents the agent read
    "arg_only_from_read",  # some argument came from data, never from the user
    "prompt_len",  # length of the user's request
    "args_len",  # serialised size of the arguments
    "prev_n",  # number of actions already taken
    # Severity, expressed as *verbs* rather than tool names, so it transfers to
    # suites whose tools the model has never seen (workspace has no send_money).
    "is_outbound",  # sends / posts / invites / reserves / shares
    "is_destructive",  # deletes / removes / cancels
    "is_mutation",  # creates / updates / reschedules / appends
    "is_readonly",  # gets / reads / searches / lists
]
CATEGORICAL_FEATURES = ["tool"]

# Hardcoded, and used by the trivial baseline this model must beat.
# These were chosen because they are the most likely to be misused.
DANGEROUS_TOOLS = {
    "send_money",
    "send_email",
    "update_password",
    "delete_file",
    "add_user_to_channel",
}

OUTBOUND_PREFIXES = ("send_", "post_", "share_", "invite_", "reserve_", "add_user")
DESTRUCTIVE_PREFIXES = ("delete_", "remove_", "cancel_")
MUTATION_PREFIXES = ("create_", "update_", "reschedule_", "append_", "schedule_")
READONLY_PREFIXES = ("get_", "read_", "search_", "list_", "check_")


def severity(tool: str) -> dict[str, float]:
    """Categorise a tool by what its verb does, not by its exact name.

    Name-based severity does not transfer: workspace and banking share no tool
    names at all, so a model trained on workspace would have no idea that
    `send_money` is dangerous. Verb categories do transfer.
    """
    tool = tool or ""
    return {
        "is_outbound": float(tool.startswith(OUTBOUND_PREFIXES)),
        "is_destructive": float(tool.startswith(DESTRUCTIVE_PREFIXES)),
        "is_mutation": float(tool.startswith(MUTATION_PREFIXES)),
        "is_readonly": float(tool.startswith(READONLY_PREFIXES)),
    }


def load_rows(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    df = pd.DataFrame(rows)
    # Real runs only: spliced traces are for training experiments, not evaluation.
    df = df[df["trace_id"].str.startswith("runs/")].copy()
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prompt_len"] = df["user_prompt"].fillna("").str.len()
    df["args_len"] = df["args"].apply(lambda a: len(json.dumps(a)))
    df["prev_n"] = df["prev_tools"].apply(len)
    df["arg_only_from_read"] = df["arg_only_from_read"].astype(float)

    # Missing provenance (no string args) is not the same as "not from read":
    # mark it explicitly, and keep the numeric column at 0.
    # If the column value is null, then we assign it -1. It's a sentinel value. In other words, we just skip it.
    df["frac_args_in_prompt"] = pd.to_numeric(
        df["frac_args_in_prompt"], errors="coerce"
    ).fillna(-1.0)
    df["frac_args_in_read"] = pd.to_numeric(
        df["frac_args_in_read"], errors="coerce"
    ).fillna(-1.0)
    sev = df["tool"].apply(severity).apply(pd.Series)
    df = pd.concat([df, sev], axis=1)
    df["group"] = df["suite"] + "::" + df["user_task_id"].astype(str)
    return df


def build_matrix(train: pd.DataFrame, test: pd.DataFrame):
    """Fit encoders on train only, then transform both. Fitting on test leaks."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train[CATEGORICAL_FEATURES])

    scaler = StandardScaler()
    scaler.fit(train[NUMERIC_FEATURES].astype(float))

    def transform(frame: pd.DataFrame) -> np.ndarray:
        cat = encoder.transform(frame[CATEGORICAL_FEATURES])
        num = scaler.transform(frame[NUMERIC_FEATURES].astype(float))
        return np.hstack([num, cat])  # type: ignore

    return transform(train), transform(test), encoder


def threshold_for_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    """Highest threshold that keeps benign false positives at or below target.

    Chosen on benign rows only - never on attacks.
    """
    ordered = np.sort(benign_scores)[::-1]
    allowed = int(len(ordered) * target_fpr)
    if allowed >= len(ordered):
        return float("-inf")
    return float(ordered[allowed])


def metrics(
    scores: np.ndarray, frame: pd.DataFrame, threshold: float
) -> tuple[float, float, float, int, int, int]:
    """Recall, benign FPR and authorized FPR at a given threshold, with counts."""
    attack = (frame["kind"] == "attack").to_numpy()
    positive = (frame["label"] == 1).to_numpy()
    benign = (frame["kind"] == "benign").to_numpy()
    authorized = (frame["kind"] == "authorized").to_numpy()

    flagged = scores >= threshold
    tp = int((flagged & attack & positive).sum())
    n_pos = int((attack & positive).sum())
    fp = int((flagged & benign).sum())
    n_ben = int(benign.sum())
    afp = int((flagged & authorized).sum())
    n_auth = int(authorized.sum())
    return (
        tp / n_pos if n_pos else float("nan"),
        fp / n_ben if n_ben else float("nan"),
        afp / n_auth if n_auth else float("nan"),
        tp,
        n_pos,
        afp,
    )


def show(label: str, scores: np.ndarray, frame: pd.DataFrame) -> None:
    """Print a model's whole operating curve, plus the rule's single point.

    A rule is a fixed operating point, not a curve, so comparing it to a model at
    one arbitrary threshold is unfair. Printing the model at several false-positive
    budgets lets the two be compared where they actually overlap.
    """
    benign_scores = scores[(frame["kind"] == "benign").to_numpy()]
    print(f"  {label}")
    for target in (0.01, 0.05, 0.10, 0.20):
        thr = threshold_for_fpr(benign_scores, target)
        rec, bfpr, afpr, tp, n_pos, afp = metrics(scores, frame, thr)
        print(
            f"      @~{target:4.0%} benign FPR:  recall={rec:6.1%} ({tp:3d}/{n_pos:3d})   "
            f"benign={bfpr:6.1%}   authorized={afpr:6.1%}"
        )

    rule = rule_scores(frame)
    rec, bfpr, afpr, tp, n_pos, afp = metrics(rule, frame, 0.5)
    print(
        f"      rule (fixed point):  recall={rec:6.1%} ({tp:3d}/{n_pos:3d})   "
        f"benign={bfpr:6.1%}   authorized={afpr:6.1%}"
    )


def rule_scores(frame: pd.DataFrame) -> np.ndarray:
    """The trivial baseline: flag known dangerous tools."""
    return frame["tool"].isin(DANGEROUS_TOOLS).astype(float).to_numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", default="../benchmark_dojo/data/rows.jsonl")
    parser.add_argument(
        "--fpr",
        type=float,
        default=0.01,
        help="target false-positive rate on benign traffic",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    path = Path(args.rows)
    if not path.exists():
        raise SystemExit(f"No rows at {path.resolve()}")
    df = add_derived(load_rows(path))
    print(
        f"rows: {len(df)}  positives: {int((df['label'] == 1).sum())}  "
        f"benign: {int((df['kind'] == 'benign').sum())}  authorized: {int((df['kind'] == 'authorized').sum())}"
    )

    # --- A. user-task-held-out split -------------------------------------
    groups = sorted(df["group"].unique())
    random.Random(args.seed).shuffle(groups)
    n_test = max(1, int(len(groups) * args.test_fraction))
    test_groups = set(groups[:n_test])
    train_a = df[~df["group"].isin(test_groups)]
    test_a = df[df["group"].isin(test_groups)]
    print(
        f"\nA. USER-TASK-HELDOUT  train={len(train_a)} rows/{train_a['group'].nunique()} tasks, "
        f"test={len(test_a)} rows/{test_a['group'].nunique()} tasks"
    )

    X_train, X_test, _ = build_matrix(train_a, test_a)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, train_a["label"])
    scores_a = model.predict_proba(X_test)[:, 1]

    show("logistic regression vs rule:", scores_a, test_a)

    # --- B. cross-suite: train workspace, test elsewhere ------------------
    print(
        "\nB. CROSS-SUITE  (train on workspace only; tool names are disjoint across suites)"
    )
    train_b = df[df["suite"] == "workspace"]
    for suite in ["banking", "slack", "travel"]:
        test_b = df[df["suite"] == suite]
        if test_b.empty or train_b.empty:
            continue
        X_tr, X_te, _ = build_matrix(train_b, test_b)
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(X_tr, train_b["label"])
        s = m.predict_proba(X_te)[:, 1]
        show(f"LR trained on workspace, tested on {suite}:", s, test_b)

    # --- what the model leans on -----------------------------------------
    print("\nTop coefficients (standardised, positive = pushes towards 'attack'):")
    coefs = model.coef_[0]
    # feature order is: numeric block, then one-hot tool block
    pairs = list(zip(NUMERIC_FEATURES, coefs[: len(NUMERIC_FEATURES)]))
    pairs.sort(key=lambda kv: -abs(kv[1]))
    for name, value in pairs:
        print(f"  {name:24s} {value:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
