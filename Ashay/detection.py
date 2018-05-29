import gdal
import gdalconst
import numpy as np
import cv2
from scipy import ndimage, misc
import math

filename = 'satellite_imagery/20180206_074601_1033/20180206_074601_1033_3B_AnalyticMS.tif'
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

srccopy = (np.uint8) (src *255) 
# print(srccopy.shape)

width,height = srccopy.shape[0], srccopy.shape[1]
image_center = (width/2, height/2)
rotation_angle = 11.1
rotation_mat = cv2.getRotationMatrix2D(image_center,rotation_angle,1)

# rotation calculates the cos and sin, taking absolutes of those.
abs_cos = abs(rotation_mat[0,0]) 
abs_sin = abs(rotation_mat[0,1])

# find the new width and height bounds
bound_w = int(height * abs_sin + width * abs_cos)
bound_h = int(height * abs_cos + width * abs_sin)

# subtract old image center (bringing image back to origo) and adding the new image center coordinates
rotation_mat[0, 2] += bound_w/2 - image_center[0]
rotation_mat[1, 2] += bound_h/2 - image_center[1]

#apply rotation
dst = cv2.warpAffine(srccopy,rotation_mat,(height,width))

#crop image to border
x_start = 520
x_end = 10000
y_start = 740
y_end = 3000

cropped = dst[y_start:y_end, x_start:x_end]

#otsu thresholding- will choose default pixel threshold
gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# apply automatic Canny edge detection using the computed median
v = np.median(cropped)
sigma = 0.33

lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

cdst = cv2.Canny(otsu, lower, upper)
cdstP = np.copy(cdst)
lines = cv2.HoughLines(cdst, 1, np.pi / 180, 250, None, 0, 0)

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
linesP = cv2.HoughLinesP(cdstP, 1, np.pi / 180, 50, None, 50, 10)
# Draw the lines
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


#scale down final product in order to view results easily
scale = 0.2
cropped_r= cv2.resize(cropped, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
cdst_r = cv2.resize(cdst, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
cdstP_r = cv2.resize(cdstP, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
otsu_r = cv2.resize(otsu, (0,0), fx=scale, fy=scale, interpolation = cv2.INTER_AREA)

cv2.imshow('src', cropped_r)
cv2.imshow('otsu', otsu_r)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst_r)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP_r)

cv2.imwrite('Source.jpg', srccopy)
cv2.imwrite('RotateCrop.jpg', cropped)
cv2.imwrite('OTSU.jpg',otsu)
cv2.imwrite('Hough.jpg',cdst)
cv2.imwrite('HoughP.jpg',cdstP)

cv2.waitKey(0)