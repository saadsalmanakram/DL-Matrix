# Saving a model
torch.save(model.state_dict(), 'model.pth')

# Loading a model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode after loading
