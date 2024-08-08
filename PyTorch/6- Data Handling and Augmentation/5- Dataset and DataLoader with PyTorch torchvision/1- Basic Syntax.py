from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Iterating through DataLoader
for images, labels in train_loader:
    print(images.shape, labels.shape)
