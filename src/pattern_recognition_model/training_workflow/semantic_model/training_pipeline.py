import argparse
import sys
import torch
import torch.optim as optim
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
    pos_preds = preds[test_df["label"] == 1]
    benign_preds = preds[test_df["kind"] == "benign"]
    auth_preds = preds[test_df["kind"] == "authorized"]

    lines = []
    lines.append(f"seed: {seed}")
    lines.extend(
        [
            str(classification_report(all_labels, preds)),
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
):
    out_dir = Path(
        "mlp_random_seeds_output/removed_step"
        if step_removed
        else "mlp_random_seeds_output/spliced"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "_".join(held_out)

    train_loader = DataLoader(
        TensorDataset(a_train, b_train, y_train), batch_size=32, shuffle=True
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
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        for epoch in range(5):  # Adjusted to 5 to prevent overfitting
            for batch_a, batch_b, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_a, batch_b), batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate and write to file
        lines = evaluate_pytorch_model(
            model, test_loader, test_df, held_out, seed, include_spliced, step_removed
        )
        file_path = (
            out_dir / f"{split_by}_{slug}_seed_{seed}_spliced_{include_spliced}.txt"
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

    train_idx = frame.index[~is_test]
    test_idx = frame.index[is_test]

    is_test_spliced = frame.loc[test_idx, "trace_id"].str.startswith(
        "runs_spliced/", na=False
    )
    test_idx = test_idx[~is_test_spliced]

    if not args.include_spliced:
        is_train_spliced = frame.loc[train_idx, "trace_id"].str.startswith(
            "runs_spliced/", na=False
        )
        train_idx = train_idx[~is_train_spliced]

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

    train_loader = DataLoader(
        TensorDataset(a_train, b_train, y_train), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(a_test, b_test, y_test), batch_size=32, shuffle=False
    )

    # 2. Initialize Model & Training Tools
    model = SemanticModel()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 5

    # 3. Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_a, batch_b, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_a, batch_b)
            loss = criterion(predictions, batch_y)
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
