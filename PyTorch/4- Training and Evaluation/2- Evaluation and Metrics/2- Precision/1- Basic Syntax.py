def precision(output, target, average='binary'):
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        tp = ((preds == 1) & (target == 1)).sum().item()
        fp = ((preds == 1) & (target == 0)).sum().item()
        precision = tp / (tp + fp)
    return precision

# Example usage
output = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
target = torch.tensor([1, 0, 1])
prec = precision(output, target)
print(f'Precision: {prec:.4f}')