import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def initialize_camera(camera_index=0):
	cap = cv2.VideoCapture(camera_index)

	# Check if the cam is opened correctly
	if not cap.isOpened():
		print("Error: Could not open camera.")
		return None

	return cap


def process_frame(frame):
	# Add your processing code here
	return frame


def compute_fov(f, w):
	return 2 * np.arctan(w / (2 * f)) * (180 / np.pi)  # Convert to degrees


def plot_fov():
	w_values = [6.0, 36.0]  # Sensor widths (mm) for smartphone and DSLR
	f_values = np.linspace(1, 100, 500)  # Focal length in mm

	theta_values = {w: compute_fov(f_values, w) for w in w_values}

	plt.figure(figsize=(8, 5))
	for w, theta in theta_values.items():
		plt.plot(f_values, theta, label=f"Sensor Width = {w}mm")

	plt.xlabel("Focal Length (mm)")
	plt.ylabel("Field of View (degrees)")
	plt.title("Field of View vs. Focal Length")
	plt.legend()
	plt.grid()
	plt.show()


def plot_projected_distance():
	dx = 10  # World space distance between x1 and x2 (arbitrary units)
	z_values = [100, 500, 1000]  # Different depths (arbitrary units)
	f_values = np.linspace(1, 100, 500)

	plt.figure(figsize=(8, 5))
	for z in z_values:
		projected_distance = (f_values * dx) / z
		plt.plot(f_values, projected_distance, label=f"Depth z = {z}")

	plt.xlabel("Focal Length (mm)")
	plt.ylabel("Projected Distance |x’_2 - x’_1|")
	plt.title("Projected Distance vs. Focal Length for Different Depths")
	plt.legend()
	plt.grid()
	plt.show()


def main():
	# Initialize the camera
	cap = initialize_camera()
	if cap is None:
		sys.exit(1)

	print("Camera feed started. Press 'q' to quit.")

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Error: Can't receive frame from camera.")
			break

		output_frame = process_frame(frame)
		cv2.imshow('Original Frame', frame)
		cv2.imshow('Processed Frame', output_frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	cap.release()
	cv2.destroyAllWindows()

	# Plot FOV and projected distance analysis
	plot_fov()
	plot_projected_distance()

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
