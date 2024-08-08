import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

class AugmentedDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

# Example transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

# Example usage
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Replace with actual paths
augmented_dataset = AugmentedDataset(image_paths, transform=transform)
dataloader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Process each batch
    pass
