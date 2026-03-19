import os
import torch
import numpy as np
from PIL import Image
from model import model, transform

dataset_path = "dataset"

features = []
image_paths = []

for img_name in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, img_name)

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        feature = model(image)

    feature = feature.squeeze().numpy()

    features.append(feature)
    image_paths.append(img_path)

np.save("features.npy", features)
np.save("paths.npy", image_paths)

print("Feature extraction completed")