import torch
from model_definition import MyModelClass  # Replace with your actual model class

model = MyModelClass()  # Initialize the model architecture
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set the model to evaluation mode if only using for inference
