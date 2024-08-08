import torch
from torchvision import models

# Version control mechanism (example)
def load_model(version='v1'):
    model = models.resnet18(pretrained=True)
    model.load_state_dict(torch.load(f'model_{version}.pth'))
    model.eval()
    return model

# Load a specific version
model_v1 = load_model('v1')
model_v2 = load_model('v2')

# Switch between models as needed
current_model = model_v1 if condition else model_v2
