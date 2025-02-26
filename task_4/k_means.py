import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

image_path = "image.jpg" 
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found! Check file path.")
    exit()

image = cv2.resize(image, (300, 200))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width, _ = image.shape

# Create a feature space of 5 dimensions (R, G, B, x, y)
X = []
for y in range(height):
    for x in range(width):
        r, g, b = image[y, x]
        X.append([r, g, b, x, y])  

X = np.array(X)

mean_shift = MeanShift(bandwidth=20)  
mean_shift.fit(X)

# Get cluster labels and cluster centers
labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_

# Replace each pixel with its cluster's mean color
segmented_pixels = cluster_centers[labels][:, :3]  # Only take RGB values
segmented_image = segmented_pixels.reshape((height, width, 3)).astype(np.uint8)

# Show images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(segmented_image)
axes[1].set_title("Segmented Image (Mean-Shift in 5D)")
axes[1].axis("off")

plt.show()