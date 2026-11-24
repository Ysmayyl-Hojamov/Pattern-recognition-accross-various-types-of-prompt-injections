import torch
import torch.nn as nn


class SemanticModel(nn.Module):
    """
    Simplest semantic model using ReLU activations and dropout to prevent overfitting.
    Branch A: Hard features (12 -> 32)
    Branch B: Embedding features (1536 -> 16)
    Fusion Head: (48 -> 1)
    """

    def __init__(self, branch_a_dim=12):
        """
        For ablation study. When step is removed, branch_a_dim will be 11 instead of 12.
        """
        super().__init__()
        self.branch_a_net = nn.Sequential(
            nn.Linear(branch_a_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.branch_b_net = nn.Sequential(
            nn.Linear(1536, 16),
            nn.ReLU(),
            nn.Dropout(0.60),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(32 + 16, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, branch_a, branch_b):
        out_a = self.branch_a_net(branch_a)
        b_flat = branch_b.view(branch_b.size(0), -1)
        out_b = self.branch_b_net(b_flat)
        combined = torch.cat([out_a, out_b], dim=1)
        return self.fusion_head(combined)
