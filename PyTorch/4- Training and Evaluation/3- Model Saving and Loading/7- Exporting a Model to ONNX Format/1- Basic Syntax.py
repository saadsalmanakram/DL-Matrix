import torch
import torch.onnx

# Assume 'model' is your PyTorch model and 'dummy_input' is a tensor with the correct input shape
dummy_input = torch.randn(1, 3, 224, 224)  # Example input
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True)
