import cv2
import sys
import numpy as np
from detectors import harris_detector, orb_detection


def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    # Check if the cam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
        
    return cap

def process_frame(frame):
    # Apply high pass filter
    frame = high_pass_filter(frame, cutoff=10)
    return frame

def high_pass_filter(frame, cutoff=40):
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
    fshift = np.fft.fftshift(f)
    
    # Get magnitude spectrum and normalize for display
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = np.log(1 + magnitude_spectrum)  # Log scale for visibility
    magnitude_spectrum = np.uint8(np.clip(magnitude_spectrum, 0, 255))
    
    return magnitude_spectrum


def main():
    # Initialize the camera
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)
  
    print("Camera feed started. Press 'q' to quit.")
    # Start capturing and processing frames
    display_frequency = False
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Error: Can't receive frame from camera.")
            break

        # Process the frame with a chosen (set) of functions
        output_frame = process_frame(frame)
        
        if display_frequency:
            output_frame = frequency_domain(output_frame)
            frame = frequency_domain(frame)

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        # Display the processed frame
        cv2.imshow('Processed Frame', output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit the application
        elif key == ord('t'):
            display_frequency = not display_frequency  # Toggle between spatial and frequency domain

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    image1 = cv2.imread('cat_1.jpg')
    image2 = cv2.imread('cat_2.jpg')
    
    result1, response1 = harris_detector(image1)
    result2, response2 = harris_detector(image2)
    orb_result, orb_keypoints = orb_detection(image1)
    orb_result2, orb_keypoints2 = orb_detection(image2)
    
    
    cv2.imshow("Harris Corners - Image 1", result1)
    cv2.imshow("Harris Corners - Image 2", result2)
    cv2.imshow("Corner Response 1", response1)
    cv2.imshow("Corner Response 2", response2)
    
    cv2.imshow("ORB Keypoints", orb_result)
    cv2.imshow("ORB Keypoints 2", orb_result2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
