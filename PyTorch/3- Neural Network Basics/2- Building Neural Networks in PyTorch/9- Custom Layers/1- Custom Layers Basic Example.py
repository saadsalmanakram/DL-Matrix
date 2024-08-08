class MyCustomLayer(nn.Module):
    def __init__(self):
        super(MyCustomLayer, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)  # Example custom activation

# Use custom layer in a model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.custom_layer = MyCustomLayer()

    def forward(self, x):
        return self.custom_layer(x)
