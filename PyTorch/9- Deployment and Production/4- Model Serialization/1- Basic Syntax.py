import torch
from torchvision import models

# Load the model
model = models.resnet18(pretrained=True)
model.eval()

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)

# Save the scripted model
scripted_model.save('scripted_model.pt')

# Load and run the scripted model
loaded_scripted_model = torch.jit.load('scripted_model.pt')
with torch.no_grad():
    output = loaded_scripted_model(torch.rand(1, 3, 224, 224))
