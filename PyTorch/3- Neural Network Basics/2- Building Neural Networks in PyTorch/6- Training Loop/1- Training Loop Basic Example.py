num_epochs = 5
for epoch in range(num_epochs):
    for data, target in dataloader:  # Assuming a DataLoader is defined
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
