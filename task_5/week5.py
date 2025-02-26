import cv2
import numpy as np
import matplotlib.pyplot as plt

image_1 = cv2.imread("image_3.jpg")
image_2 = cv2.imread("image_4.jpg")

# Detect keypoints using FAST feature detector
fast_detector = cv2.FastFeatureDetector_create(threshold=100)
keypoints = fast_detector.detect(image_1, None)

if not keypoints:
    raise ValueError("No keypoints detected. Try using a different feature detector.")

initial_points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

# Compute optical flow using Lucas-Kanade method
tracked_points, status, error = cv2.calcOpticalFlowPyrLK(image_1, image_2, initial_points, None)

if tracked_points is None or status is None:
    raise ValueError("Optical flow could not be computed. Check image quality and motion between frames.")

# Select valid points and draw lines between them 
valid_new_points = tracked_points[status.flatten() == 1]
valid_old_points = initial_points[status.flatten() == 1]

output_image = image_2.copy()
for new_point, old_point in zip(valid_new_points, valid_old_points):
    x_new, y_new = new_point.ravel()
    x_old, y_old = old_point.ravel()
    cv2.line(output_image, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
    cv2.circle(output_image, (int(x_new), int(y_new)), 3, (0, 0, 255), -1)

# difference between images
difference_image = cv2.absdiff(image_1, output_image)

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)), plt.title("Image 1")
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)), plt.title("Image 2")
plt.show()
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)), plt.title("Optical flow Image")
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(difference_image, cv2.COLOR_BGR2RGB)), plt.title("Difference Image")

plt.show()
