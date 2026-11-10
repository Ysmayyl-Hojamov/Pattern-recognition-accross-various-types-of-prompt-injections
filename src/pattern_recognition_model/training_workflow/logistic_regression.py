"""
Hard-coded functions that check the type of actions are the most vulnerable place in the code.
Might change it later. I added it only to experiment.
I will include results of logistic regression both with them and without them.

* No "spliced" runs in logistic regression. Do not include them.
"""

import argparse
import sys

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from pattern_recognition_model.utils.loaders import load_dataset

ds = load_dataset(
    "v1",
    include_spliced=False,
)

frame = pd.DataFrame(ds)

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


def is_outbound(tool_name: str) -> int:
    if "send" in tool_name.lower():
        return 1
    return 0


def is_destructive(tool_name: str) -> int:
    if "delete" in tool_name.lower():
        return 1
    return 0


def is_mutation(tool_name: str) -> int:
    if "update" in tool_name.lower():
        return 1
    return 0


def is_readonly(tool_name: str) -> int:
    if "get" in tool_name.lower() or "read" in tool_name.lower():
        return 1
    return 0


frame["is_outbound"] = frame["tool"].apply(is_outbound)
frame["is_destructive"] = frame["tool"].apply(is_destructive)
frame["is_mutation"] = frame["tool"].apply(is_mutation)
frame["is_readonly"] = frame["tool"].apply(is_readonly)


"""
Splitting into training and test sets.

A) The first split is Held-out tasks. Authorized rows don't have assigned tasks to them. Thus we just exlcude them. 

B) Cross-suite. For instance, train on workspace, and then test on banking, travel, and Slack.
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

    * held_out_tasks list is comprised of injection tasks. The held out tasks are only in the training dataset
    """
    # Exlcuded tasks. True if the injection task is in the list
    is_test = ds["injection_task_id"].isin(held_out_tasks)
    return ds[~is_test].reset_index(drop=True), ds[is_test].reset_index(drop=True)


train_df, test_df = held_out_task_split(
    ds=frame, held_out_tasks=["injection_task_3", "injection_task_4"]
)


train_df, test_df = held_out_suite_split(
    ds=frame, held_out_suites=["banking", "slack", "travel"]
)


# For more complicated functions later
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", default="suite", choices=["suite", "task"])
    return parser.parse_args(argv)


def main():
    """ """
    held_out_object = sys.argv[1]

    if held_out_object == "suite":
        train_df, test_df = held_out_suite_split(
            ds=frame, held_out_suites=["banking", "slack", "travel"]
        )
    elif held_out_object == "task":
        train_df, test_df = held_out_task_split(
            ds=frame, held_out_tasks=["injection_task_3", "injection_task_4"]
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
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, probs))


if __name__ == "__main__":
    main()
