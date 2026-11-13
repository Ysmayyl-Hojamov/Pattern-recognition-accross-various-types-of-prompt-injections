"""
A simple multi layer perceptron. It can learn more complex decision boundaries
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pattern_recognition_model.utils.loaders import load_dataset

ds = load_dataset(
    "v1",
    include_spliced=False,
)

frame = pd.DataFrame(ds)


def is_outbound(tool_name: str) -> int:
    for verb in ["send", "post", "share", "invite", "reserve", "add_user"]:
        if verb in tool_name.lower():
            return 1
    return 0


def is_destructive(tool_name: str) -> int:
    for verb in ["delete", "remove", "cancel"]:
        if verb in tool_name.lower():
            return 1
    return 0


def is_mutation(tool_name: str) -> int:
    for verb in ["update", "create", "reschedule", "append", "schedule"]:
        if verb in tool_name.lower():
            return 1
    return 0


def is_readonly(tool_name: str) -> int:
    for verb in ["get", "read", "search", "list", "check"]:
        if verb in tool_name.lower():
            return 1
    return 0


# Features that will go into the function
features = [
    "n_str_args",
    "frac_args_in_prompt",
    "frac_args_in_read",
    "arg_only_from_read",
    "step",
    "is_outbound",
    "is_destructive",
    "is_mutation",
    "is_readonly",
]

frame["is_outbound"] = frame["tool"].apply(is_outbound)
frame["is_destructive"] = frame["tool"].apply(is_destructive)
frame["is_mutation"] = frame["tool"].apply(is_mutation)
frame["is_readonly"] = frame["tool"].apply(is_readonly)


def held_out_suite_split(
    ds: pd.DataFrame,
    held_out_suites: list[str] = [
        "workspace",
        "slack",
        "travel",
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    4 suites: workspace, travel, banking, and Slack

    - Note: the banking suite is the most vulnerable suite, while workspace is the safest one.
    """
    is_test = ds["suite"].isin(held_out_suites)
    return ds[~is_test].reset_index(drop=True), ds[is_test].reset_index(drop=True)


def held_out_task_split(
    ds: pd.DataFrame,
    held_out_tasks: list[str] = [
        "injection_task_3",
        "injection_task_4",
        "injection_task_5",
        # Adding benign user tasks to see whether the model will recognize a benign session
        "user_task_9",
        "user_task_8",
        "user_task_7",
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    No "spliced" runs.
    held_out_tasks are placed exclusively in the test set.
    """
    # Authorized rows have injection attack ID in user_task_id.
    is_test = ds["injection_task_id"].isin(held_out_tasks) | ds["user_task_id"].isin(
        held_out_tasks
    )
    return ds[~is_test].reset_index(drop=True), ds[is_test].reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "split_by",
        choices=["suite", "task"],
        help="Whether to hold out by suite or by task",
    )
    parser.add_argument(
        "--held-out",
        nargs="+",
        required=True,
        help="One or more suites or tasks to hold out (e.g. --held-out workspace slack)",
    )
    parser.add_argument(
        "--seeds",
        action="store_true",
        help="Run seeds 0-9 and save reports to disk instead of plotting",
    )
    return parser.parse_args(argv)


def _format_rate(subset: np.ndarray) -> str:
    if len(subset) == 0:
        return "N/A (0/0)"
    return f"{subset.mean():.2%} ({subset.sum()}/{len(subset)})"


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    held_out: list[str],
    seed: int | None = None,
) -> list[str]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    pos_preds = preds[test_df["label"] == 1]  # type: ignore
    benign_preds = preds[test_df["kind"] == "benign"]  # type: ignore
    auth_preds = preds[test_df["kind"] == "authorized"]  # type: ignore

    lines: list[str] = []
    if seed is not None:
        lines.append(f"seed: {seed}")

    lines.extend(
        [
            str(classification_report(y_test, preds)),
            f"RECALL:         {_format_rate(pos_preds)}",
            f"Benign FPR:     {_format_rate(benign_preds)}",
            f"Authorized FPR: {_format_rate(auth_preds)}",
            f"Held out:       {', '.join(held_out)}",
        ]
    )

    # Evading the 0.5 cutoff
    fpr, tpr, _ = roc_curve(y_test, probs)
    for target in [0.01, 0.05, 0.10, 0.20]:
        i = np.searchsorted(fpr, target, side="right") - 1
        lines.append(f"FPR<={target:.0%}: recall={tpr[i]:.2f}")

    lines.append(f"ROC AUC: {roc_auc_score(y_test, probs)}")
    return lines


def run_seeds(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    held_out_object: str,
    held_out: list[str],
) -> None:
    """
    Trains the MLP with seeds 0-9 on the same split and saves one txt file per
    seed in mlp_random_seeds_output, e.g. suite_workspace_slack_seed_3.txt
    """
    out_dir = Path("mlp_random_seeds_output")
    out_dir.mkdir(exist_ok=True)
    slug = "_".join(held_out)

    X_train, y_train = train_df[features], train_df["label"]
    X_test, y_test = test_df[features], test_df["label"]

    for seed in range(10):
        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            MLPClassifier(max_iter=1000, random_state=seed),
        )
        model.fit(X_train, y_train)

        lines = evaluate_model(model, X_test, y_test, test_df, held_out, seed=seed)
        (out_dir / f"{held_out_object}_{slug}_seed_{seed}.txt").write_text(
            "\n".join(lines)
        )
        print(f"seed {seed} done")


def main() -> None:
    """
    Examples:
        uv run mlp.py suite --held-out workspace slack
        uv run mlp.py task --held-out injection_task_3 --seeds
    """
    args = parse_args()

    if args.split_by == "suite":
        train_df, test_df = held_out_suite_split(
            ds=frame, held_out_suites=args.held_out
        )
    else:
        train_df, test_df = held_out_task_split(ds=frame, held_out_tasks=args.held_out)

    if test_df.empty:
        sys.exit(f"No rows matched held-out {args.split_by}(s): {args.held_out}")

    if args.seeds:
        run_seeds(train_df, test_df, args.split_by, args.held_out)
        return

    X_train, y_train = train_df[features], train_df["label"]
    X_test, y_test = test_df[features], test_df["label"]

    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        MLPClassifier(max_iter=1000),
    )
    model.fit(X_train, y_train)

    lines = evaluate_model(model, X_test, y_test, test_df, args.held_out)
    print("\n".join(lines))

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"IN CONSIDERATION: {', '.join(args.held_out)}")
    plt.show()


if __name__ == "__main__":
    main()
