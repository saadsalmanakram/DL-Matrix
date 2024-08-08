# Common loss functions
loss_fn = nn.CrossEntropyLoss()  # For classification
loss_fn = nn.MSELoss()            # For regression

# Example of using a loss function
output = model(input_tensor)
loss = loss_fn(output, target)
