"""
Hard-coded functions that check the type of actions are the most vulnerable place in the code.
Might change it later. I added it only to experiment.
I will include results of logistic regression both with them and without them.

* No "spliced" runs in logistic regression. Do not include them.
* Compare milestones X % FPR
* See the weights of the defined features.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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


"""
Splitting into training and test sets.

A) The first split is Held-out tasks. Authorized rows don't have assigned tasks to them. Thus we just exlcude them from the testing sets for now.

B) Cross-suite. For instance, train on workspace, and then test on banking, travel, and Slack.

* Idea: I think it's worth if we pick random distributions of variables and 
then save it somewhere in the log file to then compare it to the neural network I'm about to start training. For now, I have the needed data to compare against.
"""


def held_out_suite_split(
    ds: pd.DataFrame, held_out_suites: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    4 suites: workspace, travel, banking, and Slack

    * Note: the banking suite is the most vulnerable suite. While workspace is the safest one
    """
    # True if the suite is in the list (test).
    is_test = ds["suite"].isin(held_out_suites)
    return ds[~is_test].reset_index(drop=True), ds[is_test].reset_index(drop=True)


def held_out_task_split(
    ds: pd.DataFrame, held_out_tasks: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    No "spliced" runs in logistic regression.

    * held_out_tasks list is comprised of injection tasks. The held out tasks are only in the test set
    """
    # Authorized rows have injection attack ID in user_task_id.
    is_test = ds["injection_task_id"].isin(held_out_tasks) | ds["user_task_id"].isin(
        held_out_tasks
    )
    return ds[~is_test].reset_index(drop=True), ds[is_test].reset_index(drop=True)


# For more complicated versions later
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", default="suite", choices=["suite", "task"])
    return parser.parse_args(argv)


def main():
    """
    uv run logistic_regression.py suite/task
    """
    try:
        held_out_object = sys.argv[1]
    except:
        sys.exit("Check your inputs!")

    if held_out_object == "suite":
        train_df, test_df = held_out_suite_split(
            ds=frame, held_out_suites=["banking", "slack", "travel"]
        )
    elif held_out_object == "task":
        train_df, test_df = held_out_task_split(
            ds=frame,
            # Default option. May be conflicts when switching suites.
            held_out_tasks=[
                "injection_task_3",
                "injection_task_4",
                "injection_task_5",
                # Adding benign user tasks to see whether the regression will recognize a benign session
                "user_task_9",
                "user_task_8",
                "user_task_7",
            ],
        )
    else:
        sys.exit("Invalid held_out_object! Choose either 'suite' or 'task' ")

    X_train, y_train = train_df[features], train_df["label"]
    X_test, y_test = test_df[features], test_df["label"]

    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
        ),
    )

    model.fit(X_train, y_train)

    weights = model.named_steps["logisticregression"].coef_[0]

    for i in range(0, len(features)):
        print(f"{features[i]}: {weights[i]}")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))

    pos_preds = preds[test_df["label"] == 1]  # type: ignore
    benign_preds = preds[test_df["kind"] == "benign"]  # type: ignore
    auth_preds = preds[test_df["kind"] == "authorized"]  # type: ignore

    recall = pos_preds.mean()
    benign_fpr = benign_preds.mean()
    auth_fpr = auth_preds.mean()

    print(f"RECALL:         {recall:.2%} ({pos_preds.sum()}/{len(pos_preds)})")
    print(
        f"Benign FPR:     {benign_fpr:.2%} ({benign_preds.sum()}/{len(benign_preds)})"
    )
    print(f"Authorized FPR: {auth_fpr:.2%} ({auth_preds.sum()}/{len(auth_preds)})")

    # Evading the 0.5 cutoff
    fpr, tpr, _ = roc_curve(y_test, probs)
    for target in [0.01, 0.05, 0.10, 0.20]:
        i = np.searchsorted(fpr, target, side="right") - 1
        print(f"FPR<={target:.0%}: recall={tpr[i]:.2f}")

    print("ROC AUC:", roc_auc_score(y_test, probs))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"IN CONSIDERATION: {held_out_object}")
    plt.show()


if __name__ == "__main__":
    main()
