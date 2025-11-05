import cv2
import sys
import numpy as np

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    # Check if the cam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def process_frame(frame, cutoff):
    # Apply high pass filter
    frame = high_pass_filter(frame, cutoff)
    return frame

def high_pass_filter(frame, cutoff):
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform FFT and shift the zero frequency component to the center
    f = np.fft.fft2(frame)
    fshift = np.fft.fftshift(f)

    # Create high-pass filter mask
    rows, cols = frame.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)

    # Set low-frequency region to zero (center part)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # Apply the mask to the FFT shifted image
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)

    # Perform inverse FFT to get back to spatial domain
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize the result to uint8
    img_back = np.uint8(np.clip(img_back, 0, 255))

    return img_back

def frequency_domain(frame):
    # Convert to grayscale if not already in grayscale
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform FFT and shift the zero frequency component to the center
    f = np.fft.fft2(frame)

    # Get magnitude spectrum and normalize for display
    magnitude_spectrum = np.abs(f)
    magnitude_spectrum = 5 * np.log(1 + magnitude_spectrum)  # Log scale for visibility
    magnitude_spectrum = np.uint8(np.clip(magnitude_spectrum, 0, 255))

    return magnitude_spectrum

def on_trackbar(val):
    global cutoff
    cutoff = max(1, val)

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

def apply_ransac(kp1, kp2, matches, transformation_type=cv2.RANSAC, reproj_thresh=5.0):
    # RANSAC (Random Sample Consensus) is used to filter out incorrect matches by
    # identifying the best transformation model that fits most of the data while ignoring outliers.

    # If we don't have at least 4 matches, we can't compute a valid homography.
    if len(matches) < 4:
        print("Not enough matches to apply RANSAC.")
        return [], None

    # Extract the coordinates of the matched keypoints from both images.
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix using RANSAC. This will estimate the best transformation
    # while discarding incorrect matches (outliers) that do not fit the model.
    H, mask = cv2.findHomography(pts1, pts2, method=transformation_type, ransacReprojThreshold=reproj_thresh)

    # If RANSAC fails to find a valid transformation, return empty results.
    if H is None:
        print("Homography could not be computed.")
        return [], None

    # The mask returned by findHomography marks inliers (correct matches) with 1 and outliers with 0.
    # We filter out the inliers (good matches) based on this mask.
    inliers = [m for i, m in enumerate(matches) if mask[i]]

    return inliers, H


def main():
    # Initialize the camera
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)

    print("Camera feed started. Press 'q' to quit.")
    global cutoff
    cutoff = 30
    cv2.namedWindow("Processed Frame")
    cv2.createTrackbar("Cutoff", "Processed Frame", cutoff, 100, on_trackbar)
    cv2.namedWindow("Feature Matches")

    # Capture the first frame for matching
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't receive frame from camera.")
        sys.exit(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        # Feature matching and RANSAC
        kp1, kp2, matches = feature_matching(prev_frame, frame)
        inliers, H = apply_ransac(kp1, kp2, matches)

        # Draw matches
        match_img = cv2.drawMatches(prev_frame, kp1, frame, kp2, inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Process the frame with a chosen (set) of functions
        output_frame = process_frame(frame, cutoff)

        # Display the processed frame
        cv2.imshow('Processed Frame', output_frame)

        # Display the feature matching result
        cv2.imshow("Feature Matches", match_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit the application

        # Update the previous frame for the next iteration
        prev_frame = frame.copy()

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
