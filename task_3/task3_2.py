import cv2
import sys
import numpy as np


def feature_matching(frame1, frame2):
    # ORB feature detection
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    # Create a Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between two images
    matches = bf.match(des1, des2)

    # Sort the matches based on distance (best match first)
    matches = sorted(matches, key = lambda x: x.distance)

    return kp1, kp2, matches

    
def apply_ransac(kp1, kp2, matches):
    # Convert keypoints to coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Apply RANSAC to find the best homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Use mask to filter out the outliers
    inliers = [m for i, m in enumerate(matches) if mask[i]]
    return inliers, H

def main():
    cv2.namedWindow("Feature Matches", cv2.WINDOW_NORMAL)

    image_1 = cv2.imread("image_1.jpg")
    image_2 = cv2.imread("image_2.webp")
    
    kp1, kp2, matches = feature_matching(image_1, image_2)
    inliers, H = apply_ransac(kp1, kp2, matches)

    # Draw matches
    match_img = cv2.drawMatches(image_1, kp1, image_2, kp2, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow("Feature Matches", match_img)
    
    cv2.waitKey(0)


main()