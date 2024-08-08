import torch
from torchvision import models
import logging

logging.basicConfig(level=logging.INFO)

# Load the model
model = models.resnet18(pretrained=True)
model.eval()

# Example prediction with logging
with torch.no_grad():
    input_data = torch.rand(1, 3, 224, 224)
    output = model(input_data)
    logging.info(f'Inference Output: {output}')

# Log performance metrics
logging.info(f'Model size: {torch.cuda.memory_allocated()} bytes')
