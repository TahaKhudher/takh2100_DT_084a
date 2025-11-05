import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Calibration Parameters ---
focal_length = 1733.74      # pixels
baseline_mm = 536.62        # mm
min_disp = 55
max_disp = 170
num_disp = max_disp - min_disp  # must be divisible by 16
block_size = 5

# --- Load Stereo Images (Left + Right) ---
left_img = cv2.imread("im0.png")
right_img = cv2.imread("im1.png")
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# --- StereoSGBM Setup ---
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# --- Compute Disparity Map ---
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# --- Create Valid Mask and Compute Depth Map (in mm) ---
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity)
depth_map[valid_mask] = (focal_length * baseline_mm) / disparity[valid_mask]

# --- Normalize for Display ---
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Show Results ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Disparity Map ")
plt.imshow(disp_vis, cmap='plasma')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Depth Map ")
plt.imshow(depth_vis, cmap='plasma')
plt.axis('off')

plt.tight_layout()
plt.show()
