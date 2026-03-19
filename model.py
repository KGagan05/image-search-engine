import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)
# Load pretrained model
model = models.resnet18(pretrained=True)

# Remove classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

from PIL import Image
import os

os.makedirs("dataset", exist_ok=True)

for i, (img, label) in enumerate(dataset):
    img.save(f"dataset/img_{i}.jpg")
    if i > 200:   # limit images for speed
        break