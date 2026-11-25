
from PIL import Image
from torchvision.models import ResNet18_Weights

# Get the standard preprocessing recommended
# for our pretrained ResNet18 model.
def get_image_transform():
    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()


# Load an image and convert it into a PyTorch tensor.
def preprocess_image(image_path):
    transform = get_image_transform()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)

    return image_tensor

'''
The returned tensor should have:
(3, 224, 224)
The batch dimension will be added later.
'''
