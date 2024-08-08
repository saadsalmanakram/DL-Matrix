import torch
import torch.nn as nn
import torch.optim as optim

# Define your model
model = MyModel()

# Use DataParallel to parallelize the model across available GPUs
model = nn.DataParallel(model)

# Move the model to the GPU
model = model.to('cuda')

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for inputs, labels in dataloader:
    inputs = inputs.to('cuda')
    labels = labels.to('cuda')

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
