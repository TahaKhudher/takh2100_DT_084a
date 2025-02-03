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


def main():
    # Initialize the camera
    cap = initialize_camera()
    if cap is None:
        sys.exit(1)
  
    print("Camera feed started. Press 'q' to quit.")
    # Start capturing and processing frames
    display_frequency = False
    feature_detector = False
    global cutoff
    cutoff = 30
    cv2.namedWindow("Processed Frame")
    cv2.createTrackbar("Cutoff", "Processed Frame", cutoff, 100, on_trackbar)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Error: Can't receive frame from camera.")
            break
        result1, response1 = harris_detector(frame)
        result2, response2 = orb_detection(frame)
        # Process the frame with a chosen (set) of functions
        # output_frame = process_frame(frame, cutoff)
        
        
        if display_frequency:
            output_frame = frequency_domain(output_frame)
        else:
            output_frame = process_frame(frame, cutoff)
            

        # Display the original frame
        cv2.imshow('Original Frame', frame)
        # Display the processed frame
        cv2.imshow('Processed Frame', output_frame)
        
        cv2.imshow("Harris Corners", result1)
        cv2.imshow("ORB Keypoints", result2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit the application
        elif key == ord('t'):
            display_frequency = not display_frequency  # Toggle between spatial and frequency domain
            

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
