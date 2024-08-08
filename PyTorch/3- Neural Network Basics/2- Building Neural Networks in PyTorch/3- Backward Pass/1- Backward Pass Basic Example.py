import torch.optim as optim

# Define a loss function
criterion = nn.MSELoss()

# Define an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Forward pass
output = model(input_tensor)

# Calculate loss
target = torch.randn(1, 1)  # Example target
loss = criterion(output, target)

# Zero the gradients
optimizer.zero_grad()

# Backward pass
loss.backward()

# Update weights
optimizer.step()
