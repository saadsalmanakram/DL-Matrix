import torch
import torch.nn as nn
import torch.optim as optim

# Define a Matching Network
class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)  # Output dimension

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define similarity function
def similarity(embedding, prototypes):
    return torch.mm(embedding, prototypes.t())

# Example usage
model = MatchingNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming `support_set` and `query_set` are data and labels for the few-shot task
embeddings = model(query_set)
prototypes = model(support_set)
similarities = similarity(embeddings, prototypes)
loss = nn.CrossEntropyLoss()(similarities, query_labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
