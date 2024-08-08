from torchvision import transforms

# Define augmentation pipeline
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomResizedCrop(224)
])

# Apply augmentations to an image (example with PIL image)
from PIL import Image
image = Image.open('path_to_image.jpg')
augmented_image = augmentation(image)
augmented_image.show()
