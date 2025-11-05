import numpy as np
import math

def similarity_transform(tx, ty, angle, scale=1):

    angle = math.radians(angle)
    return np.array([[scale * math.cos(angle), -scale * math.sin(angle), tx],
                     [scale * math.sin(angle), scale * math.cos(angle), ty],
                     [0, 0, 1]])

tx, ty = 3,-2
angle = -15

transformation_matrix = similarity_transform(tx, ty, angle)
print("similarity transformation matrix: \n", transformation_matrix)