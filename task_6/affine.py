import numpy as np

original_points = np.array([[0,0,1], [0,3,1], [5,3,1], [5,0,1]])
transformed_points = np.array([[1,1,1], [3,3,1], [6,3,1], [5,2,1]])

# Calculate the transformation matrix with least squares method
T_affine = np.linalg.lstsq(original_points, transformed_points, rcond=None)[0]
T_affine = T_affine.T

print("Affine Transformation matrix: \n",T_affine)