import torchvision.datasets as datasets

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)
from PIL import Image
import os

os.makedirs("dataset", exist_ok=True)

for i, (img, label) in enumerate(dataset):
    img.save(f"dataset/img_{i}.jpg")
    if i > 200:   # limit images for speed
        break