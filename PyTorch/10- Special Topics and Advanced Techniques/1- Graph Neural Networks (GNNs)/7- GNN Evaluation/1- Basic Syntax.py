model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred.eq(data.y).sum().item()
    accuracy = correct / len(data.y)
    print(f'Accuracy: {accuracy * 100:.2f}%')
