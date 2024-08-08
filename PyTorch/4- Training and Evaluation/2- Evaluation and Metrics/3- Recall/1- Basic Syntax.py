def recall(output, target, average='binary'):
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        tp = ((preds == 1) & (target == 1)).sum().item()
        fn = ((preds == 0) & (target == 1)).sum().item()
        recall = tp / (tp + fn)
    return recall

# Example usage
output = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
target = torch.tensor([1, 0, 1])
rec = recall(output, target)
print(f'Recall: {rec:.4f}')
