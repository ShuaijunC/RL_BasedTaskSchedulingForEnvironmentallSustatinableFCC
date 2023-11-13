import torch
import torch.nn as nn
import torch.nn.functional as F



class FederationCloudModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=42):
        super(FederationCloudModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x)
        return x



class FederationCloudModelTD(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=42):
        super(FederationCloudModelTD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x