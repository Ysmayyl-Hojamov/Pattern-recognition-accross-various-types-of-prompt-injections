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

from pattern_recognition_model.utils import load_dataset, DATASET_VERSION

# 1. ALWAYS load spliced data initially so the dataframe has it
INCLUDE_SPLICED = True

ds = load_dataset(
    DATASET_VERSION,
    include_spliced=INCLUDE_SPLICED,
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
    "step",  # Now uncommented; dynamically removed by the arg parser
    "is_outbound",
    "is_destructive",
    "is_mutation",
    "is_readonly",
]

frame["is_outbound"] = frame["tool"].apply(is_outbound)
frame["is_destructive"] = frame["tool"].apply(is_destructive)
frame["is_mutation"] = frame["tool"].apply(is_mutation)
frame["is_readonly"] = frame["tool"].apply(is_readonly)


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
        "--include-spliced",
        action="store_true",
        help="Include spliced runs in training set",
    )
    parser.add_argument(
        "--remove-step",
        action="store_true",
        help="Remove the step feature for ablation",
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
    include_spliced: bool,
    step_removed: bool,
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
            str(classification_report(y_test, preds, zero_division=0)),
            f"RECALL:         {_format_rate(pos_preds)}",
            f"Benign FPR:     {_format_rate(benign_preds)}",
            f"Authorized FPR: {_format_rate(auth_preds)}",
            f"Held out:       {', '.join(held_out)}",
            f"Is spliced: {include_spliced}",
            f"'step' feature removed: {step_removed}",
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
    current_features: list[str],
    include_spliced: bool,
    step_removed: bool,
) -> None:
    slug = "_".join(held_out)
    folder_name = f"held_out_{held_out_object}_{slug}_step_removed_{step_removed}_spliced_{include_spliced}"

    if step_removed:
        out_dir = Path(f"mlp_random_seeds_output/step_removed/{folder_name}")
    else:
        out_dir = Path(f"mlp_random_seeds_output/spliced_runs_data/{folder_name}")

    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = train_df[current_features], train_df["label"]
    X_test, y_test = test_df[current_features], test_df["label"]

    for seed in range(10):
        model = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            MLPClassifier(max_iter=1000, random_state=seed),
        )
        model.fit(X_train, y_train)

        lines = evaluate_model(
            model,
            X_test,
            y_test,
            test_df,
            held_out,
            include_spliced,
            step_removed,
            seed=seed,
        )

        file_path = (
            out_dir
            / f"{held_out_object}_{slug}_seed_{seed}_spliced_{include_spliced}_step_removed-{step_removed}.txt"
        )
        file_path.write_text("\n".join(lines))
        print(f"Seed {seed} complete.")


def main() -> None:
    args = parse_args()
    print(f"Holding out {args.split_by}(s): {', '.join(args.held_out)}")

    # 1. Custom Data Split (Matches PyTorch script logic)
    if args.split_by == "suite":
        is_test = frame["suite"].isin(args.held_out)
    else:
        is_test = frame["injection_task_id"].isin(args.held_out) | frame[
            "user_task_id"
        ].isin(args.held_out)

    is_spliced = frame["trace_id"].str.contains("runs_spliced", na=False)

    if args.include_spliced:
        # Force spliced runs into train, completely remove from test
        train_mask = (~is_test) | is_spliced
        test_mask = is_test & (~is_spliced)
    else:
        # Remove spliced runs entirely from everywhere
        train_mask = (~is_test) & (~is_spliced)
        test_mask = is_test & (~is_spliced)

    train_df = frame[train_mask].reset_index(drop=True)
    test_df = frame[test_mask].reset_index(drop=True)

    if test_df.empty:
        sys.exit(f"No rows matched held-out {args.split_by}(s): {args.held_out}")

    # 2. Dynamic Feature Selection
    current_features = features.copy()
    if args.remove_step:
        current_features.remove("step")

    if args.seeds:
        run_seeds(
            train_df,
            test_df,
            args.split_by,
            args.held_out,
            current_features,
            args.include_spliced,
            args.remove_step,
        )
        return

    # 3. Standard Evaluation
    X_train, y_train = train_df[current_features], train_df["label"]
    X_test, y_test = test_df[current_features], test_df["label"]

    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        MLPClassifier(max_iter=1000),
    )
    model.fit(X_train, y_train)

    lines = evaluate_model(
        model,
        X_test,
        y_test,
        test_df,
        args.held_out,
        args.include_spliced,
        args.remove_step,
    )
    print("\n".join(lines))

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"IN CONSIDERATION: {', '.join(args.held_out)}")
    plt.show()


if __name__ == "__main__":
    main()
