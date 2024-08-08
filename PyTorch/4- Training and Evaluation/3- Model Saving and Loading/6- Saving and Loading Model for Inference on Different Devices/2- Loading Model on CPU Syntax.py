import torch
from model_definition import MyModelClass  # Replace with your actual model class

device = torch.device('cpu')
model = MyModelClass()  # Initialize the model architecture
model.load_state_dict(torch.load('model_weights_gpu.pth', map_location=device))
model.eval()  # Set the model to evaluation mode
