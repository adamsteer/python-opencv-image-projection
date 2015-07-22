# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:46:03 2015

@author: adam.d.steer@gmail.com

Task: undistort an image using lens distortion coefficients
This is the first step of the process for making classified,
roughly georectified images for sea ice lead detection from
RAPPLS

Adam Steer

"""
import glob
import os
import numpy as np
import cv2

def cam_matrix(fx, fy, cx, cy):
    """ return a camera matrix given a basic set of lens parameters """
    return np.matrix([[fx, 0, cx],
		     [0, fy, cy],
		     [0, 0, 1]], dtype = 'float32' )
       
"""
Lens parameters from Photoscan
<?xml version="1.0" encoding="UTF-8"?>
<calibration>
  <projection>frame</projection>
  <width>8176</width>
  <height>6132</height>
  <fx>4.5872244511233976e+03</fx>
  <fy>4.5861338428823810e+03</fy>
  <cx>4.0538634027335138e+03</cx>
  <cy>2.9800884104632505e+03</cy>
  <skew>2.2360733669979891e+00</skew>
  <k1>-8.3717130436494567e-03</k1>
  <k2>-2.4888283244295538e-03</k2>
  <k3>1.2007722178428002e-02</k3>
  <k4>-5.3745613558169947e-03</k4>
  <p1>-1.4033639343473054e-03</p1>
  <p2>-4.6748741036082794e-04</p2>
  <date>2015-05-17T11:13:14Z</date>
</calibration>"""


fx = 4.5872244511233976e+03
fy = 4.5861338428823810e+03
cx = 4.0538634027335138e+03
cy = 2.9800884104632505e+03
skew = 2.2360733669979891e+00
k1 = -8.3717130436494567e-03
k2 = -2.4888283244295538e-03
k3 = 1.2007722178428002e-02
k4 = -5.3745613558169947e-03
p1 = -1.4033639343473054e-03
p2 = -4.6748741036082794e-04


#thefiles = ['20121023_f13_0044.jpg']

thefiles = glob.glob('../SIPEX2_f9_test/*.jpg')

img = cv2.imread(thefiles[0])


dst = np.zeros_like(img)

cam1 = cam_matrix(fx, fy, cx, cy)

d = np.array([k1, k2, p1, p2, k3])

h, w = img.shape[:2]

newCamera, roi = cv2.getOptimalNewCameraMatrix(cam1, d, (w,h), 0) 
#cam2 = cam_matrix(fx, fy, cx, cy)

"""creating a new camera matrix, as per openCV docs for camera
   calibration """

for image in thefiles:

    img = cv2.imread(image)
    print(image)

    """undistort using openCV """ 
    dst = cv2.undistort(img, cam1, d, None, newCamera)

    cv2.imwrite('../undistorted/'+image,dst)
    print('undistorted image is saved')
    dst = np.zeros_like(img)



#f, (ax0, ax1) = plt.subplots(1, 2)
#ax0.imshow(img, interpolation='nearest')
#ax1.imshow(dst, interpolation='nearest')
#plt.show()