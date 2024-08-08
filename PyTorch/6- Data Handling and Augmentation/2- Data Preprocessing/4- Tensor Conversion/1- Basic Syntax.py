from torchvision import transforms

# Convert image to tensor
to_tensor = transforms.ToTensor()

# Apply to an image (example with PIL image)
from PIL import Image
image = Image.open('path_to_image.jpg')
tensor_image = to_tensor(image)
print(tensor_image.shape)
