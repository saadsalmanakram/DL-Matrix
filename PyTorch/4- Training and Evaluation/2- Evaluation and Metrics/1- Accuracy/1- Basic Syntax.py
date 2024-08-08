import torch

def accuracy(output, target):
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
        accuracy = correct / total
    return accuracy

# Example usage
output = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
target = torch.tensor([1, 0, 1])
acc = accuracy(output, target)
print(f'Accuracy: {acc:.4f}')
