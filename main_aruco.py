import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

import aruco.charuco_board as ac


datadir = '/data/calib/'
images = np.array([os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.png')])
order = np.argsort([int(p.split('.')[-2].split('_')[-1]) for p in images])
images = images[order]

chboard = ac.ChArUcoBoard(10, 14, .02)
# chboard.to_a4_pdf('/home/user/Desktop/1.pdf')
mtx, dist, rvecs, tvecs = chboard.calibrate_camera(images[:-1])

frame = cv2.imread(images[-1])
# frame = cv2.imread('/data/coco/train2017/000000000009.jpg')
frame = cv2.undistort(frame, mtx, dist)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

corners, ids = chboard.detect_markers(gray)
frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
ac.show_tensor(frame_markers)

ret, p_rmat, p_tvec, p_rvec = chboard.estimate_pose(gray, mtx)
if ret:
    frame_pose = cv2.aruco.drawAxis(frame, mtx, dist, p_rvec, p_tvec, .1)
    ac.show_tensor(frame_pose)

a = 0
