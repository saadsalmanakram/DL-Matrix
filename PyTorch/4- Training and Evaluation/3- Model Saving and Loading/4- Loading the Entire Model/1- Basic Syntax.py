import torch

model = torch.load('entire_model.pth')
model.eval()  # Set the model to evaluation mode if only using for inference
