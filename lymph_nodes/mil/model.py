import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(
        self, feature_dim=800, hidden_dim=500, attention_dim=128, num_classes=1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes

        # --- REMOVE FOR WSI ---
        # This CNN extracts features from raw images.
        # For WSI, use a ResNet50 or similar beforehand to convert
        # patches to vectors.  DELETE this block later.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # These layers calculate the "Attention Score" for each instance.
        # It's a "Gated Attention" mechanism (Sigmoid * Tanh)
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, self.L), nn.Sigmoid()
        )
        # Aggregates the attention scores to 1 value per instance
        self.attention_weights = nn.Linear(self.L, self.K)

        # The final classifier that takes the "Weighted Bag Average" and predicts cancer/no-cancer
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * self.K, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch=1, Bag_Length, C, H, W)
        # We squeeze to remove the batch dimension since we process one bag at a time.
        x = x.squeeze(0)

        # In WSI, 'x' will already be (Bag_Length, 1024), skip this CNN step.
        H = self.feature_extractor(x)
        H = H.view(-1, self.feature_dim)  # Flatten: (Bag_Length, Feature_Dim)

        #  ATTENTION CALCULATION
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)

        # This calculates how "important" every single patch is
        A = self.attention_weights(A_V * A_U)  # Element-wise multiplication (Gating)
        A = torch.transpose(A, 1, 0)  # Transpose for matrix multiplication
        A = F.softmax(A, dim=1)  # Normalize so attention sums to 1

        #   Multiply Features (H) by Attention Weights (A)
        # If a patch is noise, A is near 0, so it contributes nothing.
        # If a patch is tumor, A is high, so it dominates the representation M.
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()  # Convert prob to 0 or 1
        # We return 'A' (Attention) for heatmap later
        return Y_prob, Y_hat, A
