import cv2
import numpy as np

img1 = cv2.imread('images/img2.jpg')
img2 = cv2.imread('images/img1.jpg')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

# FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Find matches
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe’s Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) > 10:
    # Extract matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp img1 to align with img2
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    panorama_width = width1 + width2
    panorama_height = max(height1, height2)

    # Warp img1 instead of img2 (correct alignment)
    warped_img1 = cv2.warpPerspective(img1, H, (panorama_width, panorama_height))

    # Paste img2 onto the correct position
    warped_img1[0:height2, 0:width2] = img2

    cv2.namedWindow("Stitched Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Stitched Image", warped_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("stitched_output.jpg", warped_img1)
else:
    print(f"Not enough good matches found ({len(good_matches)}/10)")
