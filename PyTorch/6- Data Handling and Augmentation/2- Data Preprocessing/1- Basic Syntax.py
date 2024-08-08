from torchvision import transforms
from PIL import Image

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an image
image = Image.open('path_to_image.jpg')
processed_image = preprocess(image)
