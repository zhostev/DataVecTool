# image_loader.py
from torchvision import datasets, transforms

class ImageLoader:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def load_images(self):
        dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
        return dataset
