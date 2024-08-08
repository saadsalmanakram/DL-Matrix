# Example pseudo-code for a basic GNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(BasicGNN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        # x: Node features
        # edge_index: Graph connectivity
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x
