import torch
import torch.nn as nn
from pattern_recognition_model.training_workflow.semantic_model import (
    frame,
    branch_a_features,
    branch_b_features,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

labels = torch.tensor(frame["label"].values, dtype=torch.float32).unsqueeze(1)


a_train, a_test, b_train, b_test, y_train, y_test = train_test_split(
    branch_a_features, branch_b_features, labels, test_size=0.2, random_state=42
)
train_dataset = TensorDataset(a_train, b_train, y_train)
test_dataset = TensorDataset(a_test, b_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SemanticModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        # BRANCH A: Hard features
        self.branch_a_net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # BRANCH B: Embedding features (Takes 1536 columns, outputs 16 numbers)
        # 4 embeddings * 384 = 1536 flat features
        self.branch_b_net = nn.Sequential(
            # Squishing 1536 inputs down to 16 outputs to prevent overfitting.
            nn.Linear(1536, 16),
            nn.ReLU(),
            nn.Dropout(0.45),
        )

        # FUSION HEAD: Glues A (32) and B (16) together (48 total inputs) -> Outputs 1
        self.fusion_head = nn.Sequential(
            nn.Linear(32 + 16, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, branch_a, branch_b):
        # 1. Process Branch A
        out_a = self.branch_a_net(branch_a)

        # 2. Process Branch B
        # First, flatten the [Batch, 4, 384] tensor into a flat [Batch, 1536] table
        b_flat = branch_b.view(branch_b.size(0), -1)
        out_b = self.branch_b_net(b_flat)

        # 3. Glue them together side-by-side (dim=1)
        combined = torch.cat([out_a, out_b], dim=1)

        # 4. Make the final prediction
        return self.fusion_head(combined)  # type: ignore


model = SemanticModel()

import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score

# Training
print("\nStarting Training...")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()  # Turns on Dropout and learning mode
    total_train_loss = 0

    for batch_a, batch_b, batch_y in train_loader:
        optimizer.zero_grad()

        predictions = model(batch_a, batch_b)
        loss = criterion(predictions, batch_y)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")


print("\nStarting Evaluation...")
model.eval()  # Turns OFF Dropout

all_predictions = []
all_real_answers = []

with torch.no_grad():
    for batch_a, batch_b, batch_y in test_loader:
        raw_scores = model(batch_a, batch_b)
        probabilities = torch.sigmoid(raw_scores)

        all_predictions.extend(probabilities.numpy())
        all_real_answers.extend(batch_y.numpy())

auc = roc_auc_score(all_real_answers, all_predictions)
binary_guesses = [1 if p > 0.5 else 0 for p in all_predictions]
acc = accuracy_score(all_real_answers, binary_guesses)

print(f"\nTest Set Accuracy: {acc * 100:.2f}%")
print(f"Test Set ROC-AUC:  {auc:.4f}")
