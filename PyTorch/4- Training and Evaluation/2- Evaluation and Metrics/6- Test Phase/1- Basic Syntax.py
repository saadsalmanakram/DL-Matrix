def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Example usage (assuming test_loader and model are defined)
test(model, test_loader, criterion)
