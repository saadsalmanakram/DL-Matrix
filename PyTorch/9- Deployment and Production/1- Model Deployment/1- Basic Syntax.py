import torch
from torchvision import models

# Load your trained model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Export the model to a file
torch.save(model.state_dict(), 'model.pth')
