model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Disable gradient computation
    for data, target in test_loader:  # Assuming a DataLoader is defined
        output = model(data)
        # Compute metrics such as accuracy
