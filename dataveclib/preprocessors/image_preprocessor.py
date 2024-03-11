# image_preprocessor.py
from torchvision import transforms

def get_preprocessing_transforms(patch_size=16):
    """Returns the preprocessing transforms for images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])
    return transform
