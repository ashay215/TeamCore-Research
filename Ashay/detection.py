import gdal
import gdalconst
import numpy as np
import cv2
from scipy import ndimage, misc
import math

filename = 'satellite_imagery/20180206_074602_1033/20180206_074602_1033_3B_AnalyticMS.tif'
filename2 = 'test.jpg'
d = gdal.Open(filename, gdalconst.GA_ReadOnly)
geoTiffDataType = gdal.GetDataTypeName(d.GetRasterBand(1).DataType)
nC = d.RasterXSize
nR = d.RasterYSize
nB = d.RasterCount

#We've read it in using gdal, and now we will convert to a numpy array

im = np.zeros((nR, nC, nB), dtype=np.uint16)
for band in range(nB):
    data = d.GetRasterBand(band+1)
    im[:,:,band] = data.ReadAsArray(0, 0, nC, nR)

#It's now a 16-bit numpy array. To display with OpenCV, normalize from 0 to 1 by dividing by max in im to get a float64 numpy array (OpenCV will either display uint8 images, 0 to 255, or it will display float64 images, 0 to 1)
src = im / im.max()

# cv2.namedWindow('src1', cv2.WINDOW_NORMAL)


srccopy = (np.uint8) (src *255) 
# ms = cv2.resize(srccopy, (1500, 800))    
# cv2.imshow('src1', ms)
# cv2.waitKey(0)
print(srccopy.shape)

 
width,height = srccopy.shape[0], srccopy.shape[1]
image_center = (width/2, height/2)
rotation_mat = cv2.getRotationMatrix2D(image_center,11.1,1)

# rotation calculates the cos and sin, taking absolutes of those.
abs_cos = abs(rotation_mat[0,0]) 
abs_sin = abs(rotation_mat[0,1])

# find the new width and height bounds
bound_w = int(height * abs_sin + width * abs_cos)
bound_h = int(height * abs_cos + width * abs_sin)

# subtract old image center (bringing image back to origo) and adding the new image center coordinates
rotation_mat[0, 2] += bound_w/2 - image_center[0]
rotation_mat[1, 2] += bound_h/2 - image_center[1]

dst = cv2.warpAffine(srccopy,rotation_mat,(height,width))

# img = cv2.resize(dst, (896,436))
img = cv2.resize(dst, None, fx=0.1, fy=0.1, interpolation = cv2.INTER_AREA)

cropped = img[74:300, 52:1000]
# cropped = img[148:600, 104:2000]
cropped_dst = dst[740:3000, 520:10000]
# img2 = cv2.resize(srccopy, (1000,500))
   
# cv2.imshow('src1', img)
# cv2.imshow('src2', cropped_dst)
# cv2.waitKey(0)

#otsu thresholding for water- will choose default pixel threshold

#rotate and crop image- can give opencv points to crop 

v = np.median(cropped)
sigma = 0.33
 
# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

cdst = cv2.Canny(cropped, lower, upper)
canny = cv2.resize(cdst, (0,0), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
cv2.imshow('src2', canny)
cv2.waitKey(0)
cdstP = np.copy(cdst)
lines = cv2.HoughLines(cdst, 1, np.pi / 180, 150, None, 0, 0)

# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# Probabilistic Line Transform
linesP = cv2.HoughLinesP(cdst, 1, np.pi / 180, 50, None, 50, 10)
# Draw the lines
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

cv2.namedWindow('src', cv2.WINDOW_NORMAL)
cv2.imshow('src', cropped)
cdst_r = cv2.resize(cdst, (0,0), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
cdstP_r = cv2.resize(cdstP, (0,0), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv2.waitKey(0)

#----------------------------------------
#OLD CODE

# import cv2
# import pandas as pd
# import numpy as np
 
# import matplotlib.pyplot as plt
# from scipy.misc import imread, imresize
# from sklearn.cluster import KMeans
# import math

# # pd.set_option('display.max_rows', 10)
# image_file = 'testimg1.tif'
# src = cv2.imread(image_file,-1) #this line causes dtype to be uint16, but then causes assertion errors with the next line commented out
# src= cv2.convertScaleAbs(src1)#this line when uncommented, causes dtype=uint8 but opens a black image without errors
# print(src.dtype)#should be uint16, but is uint8 


# cdst = cv2.Canny(src, 50, 200, None, 3)
# lines = cv2.HoughLines(cdst, 1, np.pi / 180, 150, None, 0, 0)

# # Draw the lines
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# Probabilistic Line Transform
# linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
# # Draw the lines
# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

# Show results
# cv2.imshow("Source", src)
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
# Wait and Exit
# cv2.waitKey()
 
# img1 = cv2.imread(image_file, cv2.IMREAD_ANYDEPTH)
# print(img1.dtype)
# img = cv2.convertScaleAbs(img1)
# print(img1.dtype)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('init', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.show()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(img)
# plt.show()

# # Create a SIFT Object
# sift = cv2.xfeatures2d.SIFT_create()

# pixels = []

# # Save the image to a file
# cv2.imwrite('test.jpg',img)
# print("saved")

# height, width, depth = img.shape
# for i in range(0, height):
#     for j in range(0, (width)):
#         pixels.append(img[i,j])

# np_pixels = np.array(pixels)

# # Get the Key Points from the 'gray' image pixels, this returns a numpy array
# kp = sift.detect(np_pixels, None)

# # Now we drawn the gray image and overlay the Key Points (kp)
# img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(img)
# plt.show()


# kmeans = KMeans(n_clusters=5)
# kmeans.fit(kp.tolist())
# labels = kmeans.predict(df)
# centroids = kmeans.cluster_centers_


