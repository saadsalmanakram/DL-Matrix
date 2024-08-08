from torchvision import transforms

# Rescaling and Cropping pipeline
rescale_crop = transforms.Compose([
    transforms.Resize(256),  # Resize shorter side to 256
    transforms.CenterCrop(224)  # Crop center 224x224
])

# Apply to an image (example with PIL image)
from PIL import Image
image = Image.open('path_to_image.jpg')
processed_image = rescale_crop(image)
processed_image.show()
