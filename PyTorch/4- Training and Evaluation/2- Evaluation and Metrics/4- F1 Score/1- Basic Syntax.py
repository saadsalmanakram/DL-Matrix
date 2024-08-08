def f1_score(output, target, average='binary'):
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        tp = ((preds == 1) & (target == 1)).sum().item()
        fp = ((preds == 1) & (target == 0)).sum().item()
        fn = ((preds == 0) & (target == 1)).sum().item()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Example usage
output = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
target = torch.tensor([1, 0, 1])
f1 = f1_score(output, target)
print(f'F1 Score: {f1:.4f}')
