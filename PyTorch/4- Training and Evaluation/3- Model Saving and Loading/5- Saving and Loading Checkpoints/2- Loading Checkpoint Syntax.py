import torch

# Assume 'model' is your PyTorch model and 'optimizer' is your optimizer
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.train()  # Set the model to training mode if resuming training
