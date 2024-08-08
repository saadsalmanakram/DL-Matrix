from multiprocessing import Pool
import torch
from torchvision import models

# Load the model
model = models.resnet18(pretrained=True)
model.eval()

# Function to handle prediction for a batch of inputs
def batch_predict(batch):
    with torch.no_grad():
        return model(batch)

# Simulate a batch of requests
inputs = [torch.rand(1, 3, 224, 224) for _ in range(10)]

# Use multiprocessing for scalability
with Pool(processes=4) as pool:
    results = pool.map(batch_predict, inputs)

print("Batch predictions done")
