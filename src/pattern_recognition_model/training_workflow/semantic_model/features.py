"""
"Hard" feautures for the semantic model.
These features are derived from the tool name and other attributes of the data.

1st Branch (A): hard facts and cos similarity without taking into account any text besides aligment.
2nd Branch: semantic features. derived from the embedding files
"""

from pathlib import Path

import pandas as pd

import torch


from pattern_recognition_model.utils import load_dataset, load_emb

INCLUDE_SPLICED = False

emb_prev_tools = load_emb("prev_tools")
emb_read_digest = load_emb("read_digest")
emb_tool_args = load_emb("tool_args")
emb_user_texts = load_emb("user_texts")


ds = load_dataset(
    "v1",
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
    # Step feature might be detrimental to generalization ability of the model
    # when added spliced runs as they usually
    # have more steps in them. Ablation study
    # "step",
    "is_outbound",
    "is_destructive",
    "is_mutation",
    "is_readonly",
]

frame["is_outbound"] = frame["tool"].apply(is_outbound)
frame["is_destructive"] = frame["tool"].apply(is_destructive)
frame["is_mutation"] = frame["tool"].apply(is_mutation)
frame["is_readonly"] = frame["tool"].apply(is_readonly)


def filter_spliced(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Only for testing set
    """
    is_test_spliced = test_df["trace_id"].str.startswith("runs_spliced/", na=False)
    test_df = test_df[~is_test_spliced]

    return test_df.reset_index(drop=True)


# Cosine_similarity
sim_prompt_to_action = (emb_user_texts * emb_tool_args).sum(dim=1)
sim_read_to_action = (emb_read_digest * emb_tool_args).sum(dim=1)

sim_alignment_gap = sim_prompt_to_action - sim_read_to_action

sim_features = torch.stack(
    [sim_prompt_to_action, sim_read_to_action, sim_alignment_gap], dim=1
)

hard_features_tensor = torch.tensor(frame[features].values, dtype=torch.float32)

branch_a_features = torch.cat([hard_features_tensor, sim_features], dim=1)
