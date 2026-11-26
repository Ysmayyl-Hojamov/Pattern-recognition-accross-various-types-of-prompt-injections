"""
TO-DO:
Make the code more modular to avoid duplicating it.
"""

import argparse
import sys
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve,
)
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

# Import your data
from pattern_recognition_model.training_workflow.semantic_model import (
    frame,
    branch_a_features,
    branch_b_features,
    # The model
    SemanticModel,
)

REMOVE_STEP = False
EPOCHS = 10
AUTH_WEIGHTS = {
    "medium": 5.0,
    "high": 8.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_by", choices=["suite", "task"])
    parser.add_argument("--held-out", nargs="+", required=True)
    parser.add_argument(
        "--include-spliced",
        action="store_true",
        help="Include spliced runs in test set",
    )
    parser.add_argument(
        "--remove-step",
        action="store_true",
        help="Remove the step feature for ablation",
    )
    parser.add_argument(
        "--seeds", action="store_true", help="Run 10 seeds and save to txt"
    )
    return parser.parse_args()


def _format_rate(subset: np.ndarray) -> str:
    if len(subset) == 0:
        return "N/A (0/0)"
    return f"{subset.mean():.2%} ({subset.sum()}/{len(subset)})"


def evaluate_pytorch_model(
    model, test_loader, test_df, held_out, seed, include_spliced, step_removed
):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_a, batch_b, batch_y in test_loader:
            raw_scores = model(batch_a, batch_b)
            probs = torch.sigmoid(raw_scores).squeeze(-1).numpy()

            # Handle single-item batches gracefully
            if probs.ndim == 0:
                probs = [probs.item()]

            all_probs.extend(probs)
            all_labels.extend(batch_y.squeeze(-1).numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    # Filter metrics using the exact same Pandas dataframe logic
    pos_preds = preds[test_df["label"].values == 1]
    benign_preds = preds[test_df["kind"].values == "benign"]
    auth_preds = preds[test_df["kind"].values == "authorized"]

    lines = []
    lines.append(f"seed: {seed}")
    lines.extend(
        [
            str(classification_report(all_labels, preds, zero_division=0)),
            f"RECALL:         {_format_rate(pos_preds)}",
            f"Benign FPR:     {_format_rate(benign_preds)}",
            f"Authorized FPR: {_format_rate(auth_preds)}",
            f"Held out:       {', '.join(held_out)}",
            f"Is spliced: {include_spliced}",
            f"'step' feature removed: {step_removed}",
        ]
    )

    # FPR Cutoffs
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    for target in [0.01, 0.05, 0.10, 0.20]:
        i = np.searchsorted(fpr, target, side="right") - 1
        lines.append(f"FPR<={target:.0%}: recall={tpr[i]:.2f}")

    lines.append(f"ROC AUC: {roc_auc_score(all_labels, all_probs)}")
    return lines


def run_pytorch_seeds(
    a_train,
    b_train,
    y_train,
    a_test,
    b_test,
    y_test,
    test_df,
    split_by,
    held_out,
    include_spliced,
    step_removed,
    w_train,
):
    slug = "_".join(held_out)
    # Include_spliced/!include_spliced, removed_step + spliced/ removed_step + !spliced
    folder_name = f"held_out_{split_by}_{slug}_step_removed_{step_removed}_spliced_{include_spliced}"
    if step_removed:
        out_dir = Path(f"torch_model_outputs/step_removed/{folder_name}")
    else:
        out_dir = Path(f"torch_model_outputs/spliced_runs_data/{folder_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        TensorDataset(a_train, b_train, y_train, w_train), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(a_test, b_test, y_test), batch_size=32, shuffle=False
    )

    for seed in range(10):
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize fresh model dynamically (handles 12 vs 11 features if step is removed)
        model = SemanticModel(branch_a_dim=a_train.shape[1])
        num_zeros = (y_train == 0).sum().item()
        num_ones = (y_train == 1).sum().item()

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        model.train()
        for _ in range(EPOCHS):
            for batch_a, batch_b, batch_y, batch_w in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_a, batch_b)
                loss = (criterion(predictions, batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()

        # Evaluate and write to file
        lines = evaluate_pytorch_model(
            model, test_loader, test_df, held_out, seed, include_spliced, step_removed
        )
        file_path = (
            out_dir
            / f"{split_by}_{slug}_seed_{seed}_spliced_{include_spliced}_step_removed-{step_removed}.txt"
        )
        file_path.write_text("\n".join(lines))
        print(f"Seed {seed} complete.")


def main():
    args = parse_args()
    print(f"Holding out {args.split_by}(s): {', '.join(args.held_out)}")

    # 1. Custom Data Split
    if args.split_by == "suite":
        is_test = frame["suite"].isin(args.held_out)
    else:
        is_test = frame["injection_task_id"].isin(args.held_out) | frame[
            "user_task_id"
        ].isin(args.held_out)

    is_spliced = frame["trace_id"].str.contains("runs_spliced", na=False)

    if args.include_spliced:
        # Force spliced runs into train, completely remove from test.
        # Sensitive
        train_mask = ~is_test
        test_mask = is_test & (~is_spliced)
    else:
        # Remove spliced runs entirely from everywhere
        train_mask = (~is_test) & (~is_spliced)
        test_mask = is_test & (~is_spliced)

    train_idx = frame.index[train_mask]
    test_idx = frame.index[test_mask]

    print(
        f"Train size: {len(train_idx)} | Test size: {len(test_idx)}; TOTAL: {len(test_idx) + len(train_idx)}"
    )

    if len(test_idx) == 0:
        sys.exit(f"No rows matched held-out {args.split_by}(s): {args.held_out}")

    train_idx, test_idx = train_idx.tolist(), test_idx.tolist()
    labels = torch.tensor(
        frame["label"].fillna(0).values, dtype=torch.float32
    ).unsqueeze(1)

    a_train, a_test = branch_a_features[train_idx], branch_a_features[test_idx]
    b_train, b_test = branch_b_features[train_idx], branch_b_features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    if args.remove_step:
        a_train = torch.cat([a_train[:, :4], a_train[:, 5:]], dim=1)
        a_test = torch.cat([a_test[:, :4], a_test[:, 5:]], dim=1)

    current_branch_a_dim = a_train.shape[1]

    model = SemanticModel(branch_a_dim=current_branch_a_dim)

    """
     Penalty policy; 
     #1 is for mislabeled attacks
     #2 mislabeled authorized requests  
    """

    # 1
    num_zeros = (y_train == 0).sum().item()
    num_ones = (y_train == 1).sum().item()

    sample_weights = np.ones(len(frame), dtype=np.float32)
    sample_weights[frame["label"].values == 1] = num_zeros / num_ones

    # 2
    is_auth = frame["kind"].values == "authorized"
    # Tried 8.0 11/25/26
    # Tried lowering the weight to 5.0 11/26/26
    sample_weights[is_auth] = AUTH_WEIGHTS["medium"]

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # [batch_size, 1] from [batch_size]
    w_train = torch.tensor(sample_weights[train_idx], dtype=torch.float32).unsqueeze(1)

    if args.seeds:
        run_pytorch_seeds(
            a_train,
            b_train,
            y_train,
            a_test,
            b_test,
            y_test,
            frame.loc[test_idx],
            args.split_by,
            args.held_out,
            args.include_spliced,
            args.remove_step,
            w_train,
        )
        return

    train_loader = DataLoader(
        TensorDataset(a_train, b_train, y_train, w_train), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(a_test, b_test, y_test), batch_size=32, shuffle=False
    )

    # 3. Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_a, batch_b, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_a, batch_b)
            loss = (criterion(predictions, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {total_train_loss / len(train_loader):.4f}"
        )

    # 4. Evaluation Loop
    print("\nStarting Evaluation...")
    model.eval()
    all_predictions, all_real_answers = [], []

    with torch.no_grad():
        for batch_a, batch_b, batch_y in test_loader:
            raw_scores = model(batch_a, batch_b)
            probabilities = torch.sigmoid(raw_scores)
            all_predictions.extend(probabilities.numpy())
            all_real_answers.extend(batch_y.numpy())

    auc = roc_auc_score(all_real_answers, all_predictions)
    acc = accuracy_score(
        all_real_answers, [1 if p > 0.5 else 0 for p in all_predictions]
    )

    print(f"\nTest Set Accuracy: {acc * 100:.2f}%")
    print(f"Test Set ROC-AUC:  {auc:.4f}")


if __name__ == "__main__":
    main()
