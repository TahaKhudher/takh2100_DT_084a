import cv2
import numpy as np

# implement the harris detector algorithm to detect corners in an image
def harris_detector(image, k=0.04, threshold=0.01):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    smoothed = cv2.GaussianBlur(gray, (5, 5), 1.4)
    Ix = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # Step 4: Apply Gaussian filter to the products (smoothing step)
    S_Ix2 = cv2.GaussianBlur(Ix2, (5, 5), 1.4)
    S_Iy2 = cv2.GaussianBlur(Iy2, (5, 5), 1.4)
    S_Ixy = cv2.GaussianBlur(Ixy, (5, 5), 1.4)

    # Step 5: Compute the Harris corner response function R
    det_M = S_Ix2 * S_Iy2 - S_Ixy ** 2  # determinant
    trace_M = S_Ix2 + S_Iy2  # trace
    R = det_M - k * (trace_M ** 2)  # Harris response function

    # Step 6: Thresholding the response
    # Set R above a certain threshold as corners
    corner_response = np.zeros_like(R)
    corner_response[R > threshold * R.max()] = 255

    # Step 7: Mark the corners on the original image
    result = image.copy()
    result[corner_response == 255] = [0, 0, 255]  # Red color for corners

    return result, corner_response



def orb_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()  # Initialize ORB detector
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    output = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return output, keypoints



