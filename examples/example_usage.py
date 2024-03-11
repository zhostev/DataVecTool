# /home/idea/Code/DataVecTool/examples/example_usage.py
import torch
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from dataveclib.loaders.image_loader import ImageLoader
from dataveclib.image_preprocessor import get_preprocessing_transforms
from dataveclib.image_vectorizer import ImageVectorizer

def main():
    # 图像数据集的根目录
    root_dir = '/path/to/your/image/dataset'

    # 获取预处理转换
    preprocess_transforms = get_preprocessing_transforms()

    # 实例化图像加载器
    image_loader = ImageLoader(root_dir, transform=preprocess_transforms)

    # 加载图像数据集
    dataset = image_loader.load_images()

    # 实例化图像向量化器，这里使用默认参数
    image_vectorizer = ImageVectorizer()

    # 假设我们只处理数据集中的第一张图像
    image, _ = dataset[0]  # 获取第一张图像及其标签（标签在这里未使用）
    image = image.unsqueeze(0)  # 增加一个批次维度

    # 将图像转换为嵌入向量
    vectorized_image = image_vectorizer.vectorize(image)

    print(f"Vectorized image shape: {vectorized_image.shape}")

if __name__ == "__main__":
    main()
