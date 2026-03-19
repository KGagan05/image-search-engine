import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from model import model, transform
import matplotlib.pyplot as plt

# Load saved data
features = np.load("features.npy")
paths = np.load("paths.npy")

# Load query image
query_path = "dataset/img_31.jpg"

image = Image.open(query_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    query_feature = model(image)

query_feature = query_feature.squeeze().numpy()

# Compute similarity
similarities = cosine_similarity([query_feature], features)[0]

# Get top 5 matches
top_indices = similarities.argsort()[-5:][::-1]



# Display query + results
plt.figure(figsize=(10,4))

# Show query image
plt.subplot(1,6,1)
plt.imshow(Image.open(query_path))
plt.title("Query")
plt.axis("off")

# Show similar images
for i, idx in enumerate(top_indices):
    img = Image.open(paths[idx])

    plt.subplot(1,6,i+2)
    plt.imshow(img)
    plt.title(f"{similarities[idx]:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()