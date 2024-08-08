import torch
from torch.quantization import quantize_dynamic
from torchvision import models

# Load the model
model = models.resnet18(pretrained=True)

# Apply dynamic quantization
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Save the quantized model
torch.save(quantized_model.state_dict(), 'quantized_model.pth')

# Inference with quantized model
quantized_model.eval()
with torch.no_grad()
    # Dummy input for example
    input_data = torch.rand(1, 3, 224, 224)
    output = quantized_model(input_data)
