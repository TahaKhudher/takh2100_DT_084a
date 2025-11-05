import numpy as np
import matplotlib.pyplot as plt

# Define the points
points = np.array([[2, 2], [3, 1.5], [6, 0]])  # Change these for different cases

# Define Hough transform parameters
theta_range = np.deg2rad(np.linspace(0, 180, 180))  # Theta from 0° to 180°
r_max = int(np.sqrt(10**2 + 10**2))  # Maximum possible r
r_range = np.linspace(-r_max, r_max, 2*r_max)  # Discretized r values

# Initialize accumulator
accumulator = np.zeros((len(r_range), len(theta_range)))

# Compute votes in Hough space
for x, y in points:
    for i, theta in enumerate(theta_range):
        r = x * np.cos(theta) + y * np.sin(theta)
        r_idx = np.argmin(np.abs(r_range - r))  # Find closest r index
        accumulator[r_idx, i] += 1  # Vote in accumulator

# Find the best (r, θ) with max votes (intersection in Hough space)
best_r_idx, best_theta_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
best_r = r_range[best_r_idx]
best_theta = theta_range[best_theta_idx]

# Convert (r, theta) to Cartesian (line equation)
x_vals = np.linspace(0, 7, 100)  # x range for plotting
y_vals = (best_r - x_vals * np.cos(best_theta)) / np.sin(best_theta)

# Plot original points
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], color='red', label="Given Points")
plt.plot(x_vals, y_vals, label="Best Fit Line", color="blue")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Original Points & Best Line")
plt.legend()
plt.grid()

# Plot Hough Space
plt.subplot(1, 2, 2)
plt.imshow(accumulator, extent=[0, 180, -r_max, r_max], aspect='auto', cmap='hot')
plt.colorbar(label="Votes")
plt.scatter(np.rad2deg(best_theta), best_r, color='cyan', marker='o', label="Best Intersection")
plt.xlabel("Theta (degrees)")
plt.ylabel("r (distance)")
plt.title("Hough Space (Accumulator)")
plt.legend()
plt.grid()

plt.show()
