import torch

# Assume 'model' is your PyTorch model trained on GPU
torch.save(model.state_dict(), 'model_weights_gpu.pth')
