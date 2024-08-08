def validate(model, validation_loader, criterion):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in validation_loader:
            inputs, targets = data
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total
    avg_loss = validation_loss / len(validation_loader)
    
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# Example usage (assuming validation_loader and model are defined)
validate(model, validation_loader, criterion)
