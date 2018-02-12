import cv2
import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 10)
image_file = 'giza.jpg'
 
img = cv2.imread(image_file)
plt.imshow(img)
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()

# Create a SIFT Object
sift = cv2.xfeatures2d.SIFT_create()

pixels = []

# # Now we drawn the gray image and overlay the Key Points (kp)
# img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

height, width, depth = img.shape
for i in range(0, height):
    for j in range(0, (width)):
        x = img[i,j] 
        pixels.append(x)

np_pixels = np.array(pixels)

# Get the Key Points from the 'gray' image pixels, this returns a numpy array
kp = sift.detect(np_pixels, None)


kmeans = KMeans(n_clusters=5)
# kmeans.fit(kp)

# Save the image to a file
# cv2.imwrite('sift_keypoints.jpg',img)
