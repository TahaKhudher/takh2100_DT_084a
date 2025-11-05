import matplotlib.pyplot as plt
import numpy as np
import cv2 

def add_points_to_image(points):
    img = np.zeros((800,800), dtype=np.uint8)
    points_scaled = [(int(x * 80), int(400-y*80)) for x, y in points]
    for x,y in points_scaled:
        cv2.circle(img, (x,y), 10, 255, -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

def hough_transform_lines(img):
    edges = cv2.Canny(img,50,150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 10)
    
    if lines is not None:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)
            # x0 stores the value rcos(theta)
            x0 = a*r

            # y0 stores the value rsin(theta)
            y0 = b*r
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))
            #y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

    

points_1 = [(2,2), (3,1.5), (6,0)]
points_2 = [(2,2), (5,3), (6,0)]


img_1 = add_points_to_image(points_1)
img_2 = add_points_to_image(points_2)

img_1_lines = hough_transform_lines(img_1)
img_2_lines = hough_transform_lines(img_2)

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.title("original points a")
plt.imshow(img_1)

plt.subplot(1,2,2)
plt.title("detected lines for points a")
plt.imshow(img_1_lines)
plt.show()

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.title("original points b")
plt.imshow(img_2)

plt.subplot(1,2,2)
plt.title("detected lines for points b")
plt.imshow(img_2_lines)
plt.show()




# refrences
# https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/