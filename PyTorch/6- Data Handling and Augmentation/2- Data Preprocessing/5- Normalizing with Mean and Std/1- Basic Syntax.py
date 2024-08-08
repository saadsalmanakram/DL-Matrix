from torchvision import transforms

# Define normalization
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply normalization to an image (example with PIL image)
from PIL import Image
image = Image.open('path_to_image.jpg')
normalized_image = normalize(image)
print(normalized_image.shape)
