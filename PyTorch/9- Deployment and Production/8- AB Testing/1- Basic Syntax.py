import torch
from torchvision import models
import random

# Load models
model_v1 = models.resnet18(pretrained=True)
model_v2 = models.resnet50(pretrained=True)

# Randomly select a model for A/B testing
current_model = model_v1 if random.choice([True, False]) else model_v2

# Inference with selected model
current_model.eval()
with torch.no_grad():
    output = current_model(torch.rand(1, 3, 224, 224))

print(f'A/B Test - Model used: {"ResNet18" if current_model == model_v1 else "ResNet50"}')
