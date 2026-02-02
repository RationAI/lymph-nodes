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

        # Feature extractor for MNIST (Remove this when moving to WSI)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, self.L), nn.Tanh())
        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, self.L), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.L, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * self.K, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch=1, Bag_Length, C, H, W)
        x = x.squeeze(0)

        H = self.feature_extractor(x)
        H = H.view(-1, self.feature_dim)

        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A
