import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

import aruco.charuco_board as ac
import aruco.print_cylinder as ap
import utils.io
import utils.print_paper


def show_ndarray(img):
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    dpi = 300
    # img = ap.unroll_cylinder_side(r=.04, dpi=dpi)
    # utils.print_a4.print_tensor_to_a4_pdf(img, '/home/user/Desktop/2.pdf', dpi=dpi)

    datadir = '/data/calib/'
    images = np.array([os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.HEIC')])
    order = np.argsort([int(p.split('.')[-2].split('_')[-1]) for p in images])
    images = images[order]

    chboard = ac.ChArUcoBoard(7, 10, .04)
    chboard.to_paper_pdf('/home/user/Desktop/1.pdf', paper_size='a3')
    mtx, dist, rvecs, tvecs = chboard.calibrate_camera(images[:-1])

    frame = utils.io.imread(images[-1])
    # frame = utils.io.imread('/data/coco/train2017/000000000009.jpg')
    frame = cv2.undistort(frame, mtx, dist)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cap = cv2.VideoCapture('/data/calib/4e9eaa7bdc/rgb.mp4')
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     corners, ids = chboard.detect_markers(gray)
    #     frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    #     show_ndarray(frame_markers)
    #     a = 0

    ret, p_rmat, p_tvec, p_rvec = chboard.estimate_pose(gray, mtx)
    if ret:
        frame_pose = cv2.aruco.drawAxis(frame, mtx, dist, p_rvec, p_tvec, .1)
        show_ndarray(frame_pose)

    a = 0
