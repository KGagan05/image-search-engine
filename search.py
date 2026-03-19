import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from model import model, transform

# Load saved data
features = np.load("features.npy")
paths = np.load("paths.npy")

# Load query image
query_path = "dataset/img_32.jpg"

image = Image.open(query_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    query_feature = model(image)

query_feature = query_feature.squeeze().numpy()

# Compute similarity
similarities = cosine_similarity([query_feature], features)[0]

# Get top 5 matches
top_indices = similarities.argsort()[-5:][::-1]

print("Top similar images:")

for i in top_indices:
    print(paths[i])