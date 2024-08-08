import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # Output embedding dimension

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the prototypical network loss
def prototypical_loss(embeddings, labels):
    prototypes = [embeddings[labels == i].mean(dim=0) for i in range(len(set(labels.tolist())))]
    prototypes = torch.stack(prototypes)
    distances = torch.cdist(embeddings, prototypes)
    return distances

# Example usage
model = ProtoNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming `support_set` and `query_set` are data and labels for the few-shot task
embeddings = model(support_set)
distances = prototypical_loss(embeddings, support_labels)
loss = distances.mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
