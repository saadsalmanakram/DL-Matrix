import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the MAML meta-learner
class MAML:
    def __init__(self, model, inner_lr, outer_lr):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    def meta_train(self, tasks, num_inner_steps):
        self.model.train()
        for task in tasks:
            # Create a copy of the model for the inner loop
            model_inner = copy.deepcopy(self.model)
            optimizer_inner = optim.SGD(model_inner.parameters(), lr=self.inner_lr)
            
            # Inner loop: train on the task
            for _ in range(num_inner_steps):
                x_train, y_train = task
                optimizer_inner.zero_grad()
                output = model_inner(x_train)
                loss = nn.CrossEntropyLoss()(output, y_train)
                loss.backward()
                optimizer_inner.step()

            # Meta-update
            x_val, y_val = task
            output = model_inner(x_val)
            loss = nn.CrossEntropyLoss()(output, y_val)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Example usage
model = MLP()
maml = MAML(model, inner_lr=0.01, outer_lr=0.001)
# Assuming `tasks` is a list of training and validation data for different tasks
maml.meta_train(tasks, num_inner_steps=5)
